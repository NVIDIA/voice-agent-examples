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

"""Inventory API for Nvidia Flowershop Example.

This module provides FastAPI endpoints for reading inventory status from menu.json file.
It's designed to be run independently or integrated with the main workflow.
"""

import json
import logging
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API responses
class InventoryResponse(BaseModel):
    """Model for inventory API response."""

    success: bool = Field(..., description="Whether the request was successful")
    inventory: dict[str, int] = Field(..., description="Dictionary mapping item names to quantities")


class CartResponse(BaseModel):
    """Model for cart API response."""

    success: bool = Field(..., description="Whether the request was successful")
    cart_items: dict[str, int] = Field(..., description="Dictionary mapping item names to quantities in cart")
    total_items: int = Field(..., description="Total number of items in cart")
    is_empty: bool = Field(..., description="Whether the cart is empty")
    message: str = Field(..., description="Status message")


class ClearCartResponse(BaseModel):
    """Model for clear cart API response."""

    success: bool = Field(..., description="Whether the clear operation was successful")
    message: str = Field(..., description="Status message")


# Initialize FastAPI app
app = FastAPI(
    title="Simple Inventory & Cart API",
    description="API endpoints to get current inventory status from menu.json and cart status",
    version="1.0.0",
)

# Add CORS middleware to handle cross-origin requests from the React UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class InventoryManager:
    """Manager class for inventory operations."""

    def __init__(self, file_path: str = os.getenv("MENU_PATH", "menu.json")):
        """Initialize the inventory manager.

        Args:
            file_path: Path to the menu JSON file.
        """
        self.file_path = file_path
        self._cache = None
        self._last_modified = None

    def _load_menu_data(self) -> dict:
        """Load menu data from JSON file with caching."""
        if not os.path.exists(self.file_path):
            logger.error(f"Menu file not found: {self.file_path}")
            return {}

        try:
            # Check if file was modified since last load
            current_modified = os.path.getmtime(self.file_path)
            if self._cache is None or self._last_modified != current_modified:
                with open(self.file_path) as f:
                    self._cache = json.load(f)
                self._last_modified = current_modified
                logger.info(f"Menu data loaded from {self.file_path}")

            return self._cache

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in menu file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error reading menu file: {e}")
            return {}

    def get_inventory(self) -> dict[str, int]:
        """Get current inventory data."""
        menu_data = self._load_menu_data()
        return menu_data.get("inventory", {})


# Initialize inventory manager
inventory_manager = InventoryManager()


# WebSocket Connection Manager
class ConnectionManager:
    """Manager for WebSocket connections."""

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSocket clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager
manager = ConnectionManager()


async def notify_cart_change():
    """Notify all connected clients that cart has changed."""
    message = json.dumps({"type": "cart_changed", "timestamp": time.time()})
    await manager.broadcast(message)
    logger.info("Broadcasted cart change notification")


async def notify_inventory_change():
    """Notify all connected clients that inventory has changed."""
    message = json.dumps({"type": "inventory_changed", "timestamp": time.time()})
    await manager.broadcast(message)
    logger.info("Broadcasted inventory change notification")


@app.get("/inventory", response_model=InventoryResponse)
async def get_inventory():
    """Get current inventory status - returns item names and quantities."""
    try:
        inventory = inventory_manager.get_inventory()

        if not inventory:
            return InventoryResponse(success=False, inventory={})

        return InventoryResponse(success=True, inventory=inventory)

    except Exception as e:
        logger.error(f"Error retrieving inventory: {e}")
        return InventoryResponse(success=False, inventory={})


@app.get("/cart", response_model=CartResponse)
async def get_cart():
    """Get current cart status - returns item names and quantities in cart."""
    try:
        logger.info("Cart API endpoint called")
        # Import the global cart from register module
        from . import register

        cart = register.global_cart

        cart_items = cart.get_items()
        total_items = sum(cart_items.values()) if cart_items else 0
        is_empty = cart.is_empty()

        if is_empty:
            message = "Cart is empty"
        else:
            message = f"Cart contains {total_items} items"

        return CartResponse(
            success=True, cart_items=cart_items, total_items=total_items, is_empty=is_empty, message=message
        )

    except Exception as e:
        logger.error(f"Error retrieving cart: {e}")
        return CartResponse(
            success=False, cart_items={}, total_items=0, is_empty=True, message=f"Error retrieving cart: {str(e)}"
        )


@app.post("/cart/clear", response_model=ClearCartResponse)
async def clear_cart():
    """Clear all items from the cart - returns success status and number of items cleared."""
    try:
        logger.info("Clear cart API endpoint called")
        # Import the global cart from register module
        from . import register

        cart = register.global_cart

        # Get current items count before clearing
        current_items = cart.get_items()
        items_count = sum(current_items.values()) if current_items else 0

        if items_count == 0:
            return ClearCartResponse(success=True, message="Cart is already empty")

        # Load menu for cart clearing (needed for inventory restocking)
        menu_data = inventory_manager._load_menu_data()

        if not menu_data or "menu" not in menu_data:
            # Clear cart anyway, but note inventory couldn't be restocked
            # Use empty dict as fallback menu items to avoid errors
            empty_menu = {}
            result = cart.clear(empty_menu)
            return ClearCartResponse(
                success=True,
                message=(
                    f"Cart cleared ({items_count} items), but inventory could not be "
                    "restocked due to menu unavailability"
                ),
            )

        menu_items = menu_data["menu"]

        # Clear cart using Cart class method
        result = cart.clear(menu_items)

        if result["success"]:
            # Restock inventory for all cleared items
            if "items_to_restock" in result:
                for item_name, quantity in result["items_to_restock"].items():
                    try:
                        if quantity > 0:
                            register.menu_cache.update_inventory(inventory_manager.file_path, item_name, quantity)
                    except Exception as e:
                        logger.warning(f"Could not restock {item_name}: {e}")
                        continue

            # Notify clients of cart and inventory changes
            await notify_cart_change()
            await notify_inventory_change()

            return ClearCartResponse(success=True, message=result["message"])
        else:
            return ClearCartResponse(success=False, message=f"Error clearing cart: {result['message']}")

    except Exception as e:
        logger.error(f"Error clearing cart: {e}")
        return ClearCartResponse(success=False, message=f"Error clearing cart: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time cart and inventory updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            # Echo back to confirm connection is alive
            await websocket.send_text(f"Server received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run("ui_state_api:app", host="0.0.0.0", port=8005, reload=True, log_level="info")
