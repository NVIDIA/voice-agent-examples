# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Register module for nvidia_flowershop package."""

import asyncio
import difflib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

logger = logging.getLogger(__name__)


def _apply_rewoo_executor_patch():
    from nat.agent.rewoo_agent import agent as rewoo_agent_module

    ReWOOAgentGraph = rewoo_agent_module.ReWOOAgentGraph
    _original_executor_node = ReWOOAgentGraph.executor_node

    async def _patched_executor_node(self, state):
        current_level, _ = ReWOOAgentGraph._get_current_level_status(state)
        if current_level < 0:
            return {}
        return await _original_executor_node(self, state)

    ReWOOAgentGraph.executor_node = _patched_executor_node
    logger.debug("Applied ReWOO executor patch for zero-tool plans")


_apply_rewoo_executor_patch()


# Register direct ChatResponse -> ChatResponseChunk converter to avoid NAT's indirect-conversion warning.
def _register_chat_response_to_chunk_converter():
    from nat.data_models.api_server import ChatResponse, ChatResponseChunk
    from nat.utils.type_converter import GlobalTypeConverter

    def _chat_response_to_chunk(data: ChatResponse) -> ChatResponseChunk:
        content = ""
        if data.choices and data.choices[0].message:
            content = data.choices[0].message.content or ""
        return ChatResponseChunk.from_string(content, id_=data.id, created=data.created, model=data.model)

    GlobalTypeConverter.register_converter(_chat_response_to_chunk)
    logger.debug("Registered ChatResponse -> ChatResponseChunk converter")


_register_chat_response_to_chunk_converter()

# Constants
DEFAULT_MENU_FILE = "menu.json"
MENU_UNAVAILABLE_MSG = "Error: Menu is currently unavailable"

# Global variable to track API server
_api_server_thread = None
_api_server = None
_api_server_instance = None  # Store the actual uvicorn server instance


def start_inventory_api_server(menu_file_path: str = DEFAULT_MENU_FILE, port: int = 8005, host: str = "0.0.0.0"):
    """Start the inventory API server in a background thread."""
    global _api_server_thread, _api_server, _api_server_instance

    if _api_server_thread and _api_server_thread.is_alive():
        logger.info("Inventory API server is already running")
        return

    try:
        # Import the FastAPI app from ui_state_api (now in same package)
        from .ui_state_api import app, inventory_manager

        # Update the inventory manager's file path to use absolute path
        current_dir = Path(__file__).parent  # src/nvidia_flowershop/
        aiq_agent_dir = current_dir.parent.parent  # Go up to aiq_agent directory
        abs_menu_path = os.path.abspath(os.path.join(aiq_agent_dir, menu_file_path))
        inventory_manager.file_path = abs_menu_path

        logger.info(f"Setting inventory API to use menu file: {abs_menu_path}")
        logger.info(f"File exists: {os.path.exists(abs_menu_path)}")

        # Function to run the server with proper event loop
        def run_server():
            import uvicorn

            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info(f"Starting inventory API server on {host}:{port}")
            try:
                config = uvicorn.Config(app, host=host, port=port, log_level="info")
                server = uvicorn.Server(config)
                # Store the server instance globally so we can shut it down
                global _api_server_instance
                _api_server_instance = server
                loop.run_until_complete(server.serve())
            except Exception as e:
                logger.error(f"Error starting inventory API server: {e}")
            finally:
                loop.close()

        # Start server in background thread
        _api_server_thread = threading.Thread(target=run_server, daemon=True)
        _api_server_thread.start()

        logger.info(f"Inventory API started at http://{host}:{port}/inventory")
        logger.info(f"API documentation available at http://{host}:{port}/docs")

    except Exception as e:
        logger.error(f"Failed to start inventory API server: {e}")


def stop_inventory_api_server():
    """Stop the inventory API server gracefully."""
    global _api_server_thread, _api_server, _api_server_instance

    if _api_server_thread and _api_server_thread.is_alive():
        logger.info("Stopping inventory API server...")

        try:
            # Gracefully shutdown the uvicorn server if it exists
            if _api_server_instance:
                logger.info("Shutting down uvicorn server...")
                # Signal the server to stop
                _api_server_instance.should_exit = True

                logger.info("Uvicorn server shutdown completed")
        except Exception as e:
            logger.warning(f"Error during graceful shutdown: {e}")
            logger.info("Falling back to thread termination")

        # Clear the global references
        _api_server_thread = None
        _api_server = None
        _api_server_instance = None

        logger.info("Inventory API server stopped")


# pylint: disable=unused-argument


# Menu Caching System
class MenuCache:
    """Singleton cache for menu.json to avoid repeated file I/O operations."""

    _instance = None
    _menu = None
    _last_modified = None
    _file_path = None
    # New: Inventory-specific caching
    _inventory_cache = None
    _inventory_formatted = None
    _inventory_last_update = None

    def __new__(cls):
        """Create a new instance or return existing singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_menu(self, file_path: str) -> dict:
        """Get cached menu or load from file if not cached or file changed.

        Args:
            file_path: Path to the menu JSON file

        Returns:
            Dictionary containing menu data, empty dict if file not found/invalid
        """
        # Check if we need to load/reload the menu
        if self._should_reload_menu(file_path):
            self._load_menu(file_path)
        return self._menu or {}

    def get_inventory(self, file_path: str) -> str:
        """Get pre-formatted inventory string for faster retrieval.

        Args:
            file_path: Path to the menu JSON file

        Returns:
            Pre-formatted inventory string
        """
        # Check if inventory cache is valid
        if self._is_inventory_cache_valid(file_path):
            return self._inventory_formatted or "Inventory is currently unavailable. Please try again later."

        # Update inventory cache
        self._update_inventory_cache(file_path)
        return self._inventory_formatted or "Inventory is currently unavailable. Please try again later."

    def _is_inventory_cache_valid(self, file_path: str) -> bool:
        """Check if inventory cache is still valid."""
        if self._inventory_cache is None or self._inventory_formatted is None:
            return False

        if self._file_path != file_path or not os.path.exists(file_path):
            return False

        try:
            current_modified = os.path.getmtime(file_path)
            return self._inventory_last_update == current_modified
        except OSError:
            return False

    def _update_inventory_cache(self, file_path: str):
        """Update inventory-specific cache with pre-formatted strings."""
        menu_data = self.get_menu(file_path)
        if not menu_data or "inventory" not in menu_data:
            self._inventory_cache = {}
            self._inventory_formatted = "No inventory data found in the menu file."
            self._inventory_last_update = None
            return

        self._inventory_cache = menu_data["inventory"].copy()

        # Pre-compute formatted inventory string
        formatted_lines = ["INVENTORY:", ""]
        for item, quantity in self._inventory_cache.items():
            status = "in stock" if quantity > 0 else "out of stock"
            formatted_lines.append(f"{item}: {quantity} {status}")

        self._inventory_formatted = "\n".join(formatted_lines)

        # Update timestamp
        try:
            self._inventory_last_update = os.path.getmtime(file_path)
        except OSError:
            self._inventory_last_update = None

    def get_item_stock(self, file_path: str, items: str | list[str]) -> int | dict[str, int]:
        """Quick inventory check for one or more items without full cache reload.

        Args:
            file_path: Path to the menu JSON file
            items: Single item name (str) or list of item names (List[str])

        Returns:
            - If items is str: Current inventory quantity (int), or -1 if item not found
            - If items is List[str]: Dictionary mapping item names to quantities (-1 if not found)
        """
        # Ensure inventory cache is valid
        if not self._is_inventory_cache_valid(file_path):
            self._update_inventory_cache(file_path)

        if not self._inventory_cache:
            if isinstance(items, str):
                return -1
            else:
                return {name: -1 for name in items}

        # Handle single item case
        if isinstance(items, str):
            return self._inventory_cache.get(items, -1)

        # Handle multiple items case
        return {name: self._inventory_cache.get(name, -1) for name in items}

    def get_low_stock_items(self, file_path: str, threshold: int = 5) -> dict:
        """Get items with low stock levels efficiently.

        Args:
            file_path: Path to the menu JSON file
            threshold: Stock level threshold (default: 5)

        Returns:
            Dictionary of items with stock <= threshold
        """
        # Ensure inventory cache is valid
        if not self._is_inventory_cache_valid(file_path):
            self._update_inventory_cache(file_path)

        if not self._inventory_cache:
            return {}

        return {name: qty for name, qty in self._inventory_cache.items() if qty <= threshold}

    def _should_reload_menu(self, file_path: str) -> bool:
        """Check if menu should be reloaded."""
        if self._menu is None or self._file_path != file_path:
            return True

        if not os.path.exists(file_path):
            return False

        try:
            current_modified = os.path.getmtime(file_path)
            return self._last_modified != current_modified
        except OSError:
            return False

    def _load_menu(self, file_path: str):
        """Load menu from file and update cache."""
        self._file_path = file_path

        if not os.path.exists(file_path):
            logger.warning(f"Menu file not found: {file_path}")
            self._clear_all_caches()
            return

        try:
            self._last_modified = os.path.getmtime(file_path)
            with open(file_path) as f:
                self._menu = json.load(f)
            logger.info(f"Menu loaded successfully from {file_path}")
            self._clear_inventory_cache()

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading menu file {file_path}: {e}")
            self._clear_all_caches()

    def _clear_inventory_cache(self):
        """Clear only inventory cache."""
        self._inventory_cache = None
        self._inventory_formatted = None
        self._inventory_last_update = None

    def _clear_all_caches(self):
        """Clear all caches."""
        self._menu = {}
        self._last_modified = None
        self._clear_inventory_cache()

    def update_inventory(self, file_path: str, item_name: str, quantity_change: int) -> bool:
        """Update inventory for an item by changing the quantity.

        Args:
            file_path: Path to the menu JSON file
            item_name: Name of the item to update
            quantity_change: Amount to change (positive to add, negative to remove)

        Returns:
            True if update was successful, False otherwise
        """
        menu_data = self.get_menu(file_path)
        if not menu_data or "inventory" not in menu_data or item_name not in menu_data["inventory"]:
            return False

        current_quantity = menu_data["inventory"][item_name]
        new_quantity = current_quantity + quantity_change

        if new_quantity < 0:
            return False

        menu_data["inventory"][item_name] = new_quantity

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=os.path.dirname(file_path), delete=False, suffix=".tmp"
            ) as tmp_file:
                json.dump(menu_data, tmp_file, indent=2)
                tmp_file_path = tmp_file.name

            os.replace(tmp_file_path, file_path)

            self._menu = menu_data
            self._last_modified = os.path.getmtime(file_path)

            if self._inventory_cache is not None:
                self._inventory_cache[item_name] = new_quantity
                self._update_inventory_cache(file_path)

            return True
        except Exception as e:
            logger.error(f"Error updating inventory file {file_path}: {e}")
            return False


# Create global menu cache instance
menu_cache = MenuCache()

# Start the inventory API server as soon as MenuCache is initialized
start_inventory_api_server()


class Cart:
    """Shopping cart class for managing cart operations."""

    def __init__(self):
        """Initialize an empty shopping cart."""
        self._items = {}  # {item_name: quantity}

    def add_item(self, item_name: str, quantity: int, price: float) -> dict:
        """Add an item to the cart.

        Args:
            item_name: Name of the item
            quantity: Quantity to add
            price: Price per item

        Returns:
            Dict with success status, message, and updated cart
        """
        if quantity <= 0:
            return {"success": False, "message": "Quantity must be positive", "cart": self.get_cart_json()}

        current_quantity = self._items.get(item_name, 0)
        new_quantity = current_quantity + quantity
        self._items[item_name] = new_quantity

        total_price = quantity * price
        return {
            "success": True,
            "message": f"Added {quantity} {item_name} worth ${total_price:.2f} to cart",
            "cart": self.get_cart_json(),
            "quantity_added": quantity,
            "total_price": total_price,
        }

    def remove_item(self, item_name: str, quantity: int, price: float) -> dict:
        """Remove an item from the cart.

        Args:
            item_name: Name of the item
            quantity: Quantity to remove
            price: Price per item

        Returns:
            Dict with success status, message, and updated cart
        """
        if item_name not in self._items:
            return {
                "success": False,
                "message": f"Item '{item_name}' not found in cart",
                "cart": self.get_cart_json(),
            }

        current_quantity = self._items[item_name]
        remove_quantity = min(quantity, current_quantity)

        if remove_quantity <= 0:
            return {"success": False, "message": "No quantity to remove", "cart": self.get_cart_json()}

        new_quantity = current_quantity - remove_quantity
        if new_quantity <= 0:
            del self._items[item_name]
        else:
            self._items[item_name] = new_quantity

        total_price = remove_quantity * price
        return {
            "success": True,
            "message": f"Removed {remove_quantity} {item_name} worth ${total_price:.2f} from cart",
            "cart": self.get_cart_json(),
            "quantity_removed": remove_quantity,
            "total_price": total_price,
        }

    def clear(self, menu_items: dict) -> dict:
        """Clear the entire cart.

        Args:
            menu_items: Dictionary of menu items with prices

        Returns:
            Dict with success status, message, and cleared cart
        """
        if not self._items:
            return {"success": True, "message": "Cart is already empty", "cart": self.get_cart_json()}

        total_quantity = sum(self._items.values())
        total_price = sum(qty * menu_items.get(item, {}).get("price", 0) for item, qty in self._items.items())

        # Store items for restocking before clearing
        items_to_restock = self._items.copy()

        self._items.clear()

        return {
            "success": True,
            "message": f"Cleared cart with total quantity {total_quantity} worth ${total_price:.2f}",
            "cart": self.get_cart_json(),
            "total_quantity": total_quantity,
            "total_price": total_price,
            "items_to_restock": items_to_restock,
        }

    def view(self, menu_items: dict) -> dict:
        """View the current cart contents.

        Args:
            menu_items: Dictionary of menu items with prices

        Returns:
            Dict with cart information
        """
        if not self._items:
            return {
                "success": True,
                "message": "Your cart is empty",
                "cart": self.get_cart_json(),
                "total_quantity": 0,
                "total_price": 0.0,
            }

        total_quantity = sum(self._items.values())
        total_price = sum(qty * menu_items.get(item, {}).get("price", 0) for item, qty in self._items.items())

        return {
            "success": True,
            "message": f"Your cart has {total_quantity} items worth ${total_price:.2f}",
            "cart": self.get_cart_json(),
            "total_quantity": total_quantity,
            "total_price": total_price,
        }

    def get_cart_json(self) -> str:
        """Get cart contents as JSON string."""
        return json.dumps(self._items)

    def is_empty(self) -> bool:
        """Check if cart is empty."""
        return len(self._items) == 0

    def get_items(self) -> dict:
        """Get cart items dictionary."""
        return self._items.copy()

    def has_item(self, item_name: str) -> bool:
        """Check if cart contains an item."""
        return item_name in self._items

    def get_item_quantity(self, item_name: str) -> int:
        """Get quantity of a specific item in cart."""
        return self._items.get(item_name, 0)


# Global cart instances - could be extended for per-user cart management
global_cart = Cart()

# Dictionary to store per-user carts (demonstration of user-specific state management)
user_carts = {}  # Format: {user_id: Cart()}


def get_user_cart(user_id: str) -> Cart:
    """Get or create a cart for a specific user.

    Args:
        user_id: The user's session/identification string

    Returns:
        Cart instance for the user
    """
    if user_id not in user_carts:
        user_carts[user_id] = Cart()
    return user_carts[user_id]


# Helper functions to reduce code duplication
def get_user_id_from_context() -> str | None:
    """Extract user ID from context metadata."""
    try:
        context = Context.get()
        if hasattr(context, "metadata") and context.metadata.cookies:
            return context.metadata.cookies.get("nat-session")
    except Exception:
        pass
    return None


def format_cart_summary(cart_json: str, menu_items: dict = None) -> str:
    """Format cart items into a readable summary."""
    cart_items = json.loads(cart_json)
    if not cart_items:
        return "empty"

    items_list = []
    for item, qty in cart_items.items():
        if menu_items:
            price = menu_items.get(item, {}).get("price", 0)
            total_price = qty * price
            items_list.append(f"{qty}x {item} at ${price:.2f} each (${total_price:.2f} total)")
        else:
            items_list.append(f"{qty}x {item}")

    return ", ".join(items_list)


def find_best_match(search_term: str, candidates: list[str], threshold: float = 0.5) -> tuple[str | None, float]:
    """Find the best fuzzy match for a search term in a list of candidates.

    Args:
        search_term: The term to search for
        candidates: List of candidate strings to match against
        threshold: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        Tuple of (best_match, similarity_score) or (None, 0.0) if no match above threshold
    """
    best_match = None
    best_similarity = 0.0

    for candidate in candidates:
        similarity = difflib.SequenceMatcher(None, search_term.lower(), candidate.lower()).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = candidate

    return (best_match, best_similarity) if best_match and best_similarity > threshold else (None, 0.0)


async def notify_websocket_changes():
    """Send WebSocket notifications for cart and inventory changes."""
    try:
        from .ui_state_api import notify_cart_change, notify_inventory_change

        await asyncio.wait_for(
            asyncio.gather(notify_cart_change(), notify_inventory_change(), return_exceptions=True), timeout=3.0
        )
    except TimeoutError:
        logger.warning("WebSocket notifications timed out after 3 seconds")
    except Exception as e:
        logger.warning(f"Could not send WebSocket notifications: {e}")


class GetMenuToolConfig(FunctionBaseConfig, name="get_menu"):
    """Configuration for the get_menu tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=GetMenuToolConfig)
async def get_menu(config: GetMenuToolConfig, builder: Builder):
    """Register the get_menu function."""

    async def _get_menu(unused_input: str) -> str:
        menu_data = menu_cache.get_menu(config.file_path)
        if not menu_data:
            return "Menu is currently unavailable. Please try again later."

        menu_items = menu_data.get("menu", menu_data)

        formatted_menu = "MENU:\n\n"
        for item in menu_items:
            formatted_menu += f"{item}\n"

        return formatted_menu.strip()

    # Create a Generic NAT Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _get_menu,
        description=(
            "This is a tool to get the menu. This tool doesn't require input. "
            "It returns the list of menu item names only."
        ),
    )


class GetInventoryToolConfig(FunctionBaseConfig, name="get_inventory"):
    """Configuration for the get_inventory tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=GetInventoryToolConfig)
async def get_inventory(config: GetInventoryToolConfig, builder: Builder):
    """Register the get_inventory function."""

    async def _get_inventory(unused: str) -> str:
        # Use the optimized inventory cache for instant retrieval
        return menu_cache.get_inventory(config.file_path)

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _get_inventory,
        description=(
            "This is a tool to get the current inventory. "
            "It returns the stock levels for all items without requiring any input."
        ),
    )


class CheckItemStockToolConfig(FunctionBaseConfig, name="check_item_stock"):
    """Configuration for the check_item_stock tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=CheckItemStockToolConfig)
async def check_item_stock(config: CheckItemStockToolConfig, builder: Builder):
    """Register the check_item_stock function."""

    async def _check_item_stock(item_name: str) -> str:
        """Check stock for a specific item quickly."""
        stock_level = menu_cache.get_item_stock(config.file_path, item_name)

        if stock_level == -1:
            return f"Item '{item_name}' not found in inventory"
        elif stock_level == 0:
            return f"{item_name}: 0 out of stock"
        elif stock_level <= 5:
            return f"{item_name}: {stock_level} in stock (LOW STOCK WARNING)"
        else:
            return f"{item_name}: {stock_level} in stock"

    yield FunctionInfo.from_fn(
        _check_item_stock,
        description=(
            "This tool quickly checks the stock level for a specific item. "
            "It takes an item name as input and returns just that item's stock status. "
            "This is much faster than getting the full inventory when you only need one item."
        ),
    )


class GetPriceToolConfig(FunctionBaseConfig, name="get_price"):
    """Configuration for the get_price tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=GetPriceToolConfig)
async def get_price(config: GetPriceToolConfig, builder: Builder):
    """Register the get_price function."""

    async def _get_price(text: str) -> str:
        menu_data = menu_cache.get_menu(config.file_path)
        if not menu_data:
            return MENU_UNAVAILABLE_MSG

        menu_items = menu_data.get("menu", menu_data)

        # First try exact match (case-insensitive)
        for item, item_data in menu_items.items():
            if item.lower() == text.lower():
                return f"The price for {item} is ${item_data['price']:.2f}"

        # If no exact match, find most similar item
        best_match, _ = find_best_match(text, list(menu_items.keys()))
        if best_match:
            return f"Item '{text}' not found. Did you mean '{best_match}'?"

        return f"Item '{text}' not found in menu. Please check the menu for available items."

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _get_price,
        description=(
            "This is a tool to get the price of an item. "
            "It takes an item name as input and returns the price. "
            "If exact match is not found, it suggests the most similar item."
        ),
    )


class AddToCartToolConfig(FunctionBaseConfig, name="add_to_cart"):
    """Configuration for the add_to_cart tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=AddToCartToolConfig)
async def add_to_cart(config: AddToCartToolConfig, builder: Builder):
    """Register the add_to_cart function."""

    async def _add_to_cart(text: str) -> str:
        user_id = get_user_id_from_context()

        # Parse input to extract item name and quantity
        quantity_match = re.search(r"(\d+)\s+(.+)", text.lower())
        if quantity_match:
            quantity = int(quantity_match.group(1))
            item_name = quantity_match.group(2).strip()
        else:
            quantity = 1
            item_name = text.strip()

        menu_data = menu_cache.get_menu(config.file_path)
        if not menu_data:
            return MENU_UNAVAILABLE_MSG

        menu_items = menu_data.get("menu", menu_data)
        user_cart = get_user_cart(user_id) if user_id else global_cart

        # Search for item in menu (case-insensitive)
        found_item_name = None
        item_price = None

        for item, item_data in menu_items.items():
            if item.lower() == item_name.lower():
                found_item_name = item
                item_price = item_data["price"]
                break

        # If no exact match, try fuzzy matching
        if found_item_name is None:
            best_match, _ = find_best_match(item_name, list(menu_items.keys()))
            if best_match:
                return (
                    f"Item '{item_name}' not found. Did you mean '{best_match}'? "
                    f"Please try again with the correct name."
                )

            cart_summary = format_cart_summary(user_cart.get_cart_json())
            return (
                f"Error: '{item_name}' not found in menu. "
                f"Please check the menu for available items. "
                f"Cart unchanged, contains: {cart_summary}"
            )

        # Check available inventory before adding to cart
        available_stock = menu_cache.get_item_stock(config.file_path, found_item_name)

        if available_stock == -1:
            cart_summary = format_cart_summary(user_cart.get_cart_json())
            return f"Error: Could not check inventory for '{found_item_name}'. Cart contains: {cart_summary}"

        if available_stock == 0:
            cart_summary = format_cart_summary(user_cart.get_cart_json())
            return f"Sorry, '{found_item_name}' is out of stock. Cart contains: {cart_summary}"

        # Calculate actual quantity to add (minimum of requested vs available)
        actual_quantity = min(quantity, available_stock)

        # Add available quantity to cart using Cart class
        result = user_cart.add_item(found_item_name, actual_quantity, item_price)

        if result["success"]:
            menu_cache.update_inventory(config.file_path, found_item_name, -actual_quantity)
            await notify_websocket_changes()

            cart_summary = format_cart_summary(result["cart"], menu_items)

            # Notify user about quantity adjustment if needed
            if actual_quantity < quantity:
                shortage = quantity - actual_quantity
                return (
                    f"{result['message']} Note: Only {actual_quantity} of {quantity} "
                    f"requested items were available. {shortage} items were not available. "
                    f"Updated cart contains: {cart_summary}"
                )
            else:
                return f"{result['message']}. Updated cart contains: {cart_summary}"
        else:
            return f"Error adding item to cart: {result['message']}"

    yield FunctionInfo.from_fn(
        _add_to_cart,
        description=(
            "Adds an item to the cart. Input: text (item name or 'quantity item'). "
            "On update, returns: 'Added {quantity} {item} worth {total_price} to cart. Updated cart: {cart_status}'. "
            "If no update occurs (e.g., item not found), returns a brief message."
        ),
    )


class RemoveFromCartToolConfig(FunctionBaseConfig, name="remove_from_cart"):
    """Configuration for the remove_from_cart tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=RemoveFromCartToolConfig)
async def remove_from_cart(config: RemoveFromCartToolConfig, builder: Builder):
    """Register the remove_from_cart function."""

    async def _remove_from_cart(text: str) -> str:
        user_id = get_user_id_from_context()

        # Parse input to extract item name and quantity
        quantity_match = re.search(r"(\d+)\s+(.+)", text.lower())
        if quantity_match:
            quantity = int(quantity_match.group(1))
            item_name = quantity_match.group(2).strip()
        else:
            quantity = 1
            item_name = text.strip()

        menu_data = menu_cache.get_menu(config.file_path)
        if not menu_data:
            return MENU_UNAVAILABLE_MSG

        menu_items = menu_data.get("menu", menu_data)
        user_cart = get_user_cart(user_id) if user_id else global_cart

        # Find the actual item name in the cart (case-insensitive); try fuzzy match if needed
        cart_items = user_cart.get_items()
        actual_item_name = None

        for existing_name in cart_items:
            if existing_name.lower() == item_name.lower():
                actual_item_name = existing_name
                break

        if actual_item_name is None:
            if cart_items:
                best_match, _ = find_best_match(item_name, list(cart_items.keys()))
                if best_match:
                    cart_summary = format_cart_summary(user_cart.get_cart_json())
                    return (
                        f"Item '{item_name}' not found in cart. "
                        f"Did you mean '{best_match}'? "
                        f"Cart unchanged, contains: {cart_summary}"
                    )

            cart_summary = format_cart_summary(user_cart.get_cart_json())
            return f"Item '{item_name}' not found in cart. Cart unchanged, contains: {cart_summary}"

        item_price = menu_items.get(actual_item_name, {}).get("price", 0)
        result = user_cart.remove_item(actual_item_name, quantity, item_price)

        if result["success"]:
            menu_cache.update_inventory(config.file_path, actual_item_name, result["quantity_removed"])
            await notify_websocket_changes()

            cart_summary = format_cart_summary(result["cart"])
            return f"{result['message']}. Updated cart contains: {cart_summary}"
        else:
            return f"Error removing item from cart: {result['message']}"

    yield FunctionInfo.from_fn(
        _remove_from_cart,
        description=(
            "This tool removes an item from the shopping cart. "
            "Input: text (item name or 'quantity item'). "
            "On update, returns: 'Removed {quantity} {item} worth {total_price} from cart. "
            "Updated cart: {cart_status}'. "
            "If item is not in cart, returns a brief message."
        ),
    )


class ClearCartToolConfig(FunctionBaseConfig, name="clear_cart"):
    """Configuration for the clear_cart tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=ClearCartToolConfig)
async def clear_cart(config: ClearCartToolConfig, builder: Builder):
    """Register the clear_cart function."""

    async def _clear_cart(unused: str) -> str:
        user_id = get_user_id_from_context()

        menu_data = menu_cache.get_menu(config.file_path)
        if not menu_data:
            return MENU_UNAVAILABLE_MSG

        menu_items = menu_data.get("menu", menu_data)
        user_cart = get_user_cart(user_id) if user_id else global_cart

        result = user_cart.clear(menu_items)

        if result["success"]:
            # Restock inventory for all items that were cleared
            if "items_to_restock" in result:
                for item_name, quantity in result["items_to_restock"].items():
                    if quantity > 0:
                        try:
                            menu_cache.update_inventory(config.file_path, item_name, quantity)
                        except Exception as e:
                            logger.warning(f"Failed to restock {item_name}: {e}")

            await notify_websocket_changes()
            return result["message"]
        else:
            return f"Error clearing cart: {result['message']}"

    yield FunctionInfo.from_fn(
        _clear_cart,
        description=(
            "Clears the cart. This tool doesn't require any meaningful input. "
            "Restocks inventory for all items found in the cart, then returns: "
            "'Cleared cart with total quantity {quantity} worth ${price}'."
        ),
    )


class ViewCartToolConfig(FunctionBaseConfig, name="view_cart"):
    """Configuration for the view_cart tool."""

    file_path: str = Field(default=DEFAULT_MENU_FILE, description="Path to the menu JSON file")


@register_function(config_type=ViewCartToolConfig)
async def view_cart(config: ViewCartToolConfig, builder: Builder):
    """Register the view_cart function."""

    async def _view_cart(unused_input: str) -> str:
        user_id = get_user_id_from_context()

        menu_data = menu_cache.get_menu(config.file_path)
        if not menu_data:
            return MENU_UNAVAILABLE_MSG

        menu_items = menu_data.get("menu", menu_data)
        user_cart = get_user_cart(user_id) if user_id else global_cart

        result = user_cart.view(menu_items)
        cart_items = json.loads(result["cart"])

        if cart_items:
            items_description = []
            for item_name, quantity in cart_items.items():
                price = menu_items.get(item_name, {}).get("price", 0)
                total_item_price = quantity * price
                items_description.append(f"{quantity} x {item_name} at ${price:.2f} each = ${total_item_price:.2f}")
            cart_details = ", ".join(items_description)
            return f"{result['message']}. Cart contains: {cart_details}"
        else:
            return result["message"]

    yield FunctionInfo.from_fn(
        _view_cart,
        description=(
            "Shows the current shopping cart. This tool doesn't require input. "
            "Returns a human-readable message with cart contents, showing each item's quantity, "
            "unit price, and total price."
        ),
    )
