# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NVIDIA Flowershop NeMo Agent Toolkit Package.

This package provides a complete example of an end-to-end agentic workflow using the
NeMo Agent Toolkit, showcasing a flower shop assistant with custom function registration,
API deployment, and Phoenix tracing capabilities.

Key Features:
    - Custom Function Registration: Demonstrates custom function creation using the
      NeMo Agent toolkit registration system
    - Flower Shop Assistant: Interactive menu browsing, pricing, and cart management
      functionality
    - ReWOO Agent: Implements Reasoning Without Observation, separating planning, execution, and solving phases
    - RESTful API Deployment: Production-ready API deployment using `nat serve`
    - Phoenix Tracing: Comprehensive observability with Phoenix tracing and monitoring
    - Workflow Profiling: Built-in profiling capabilities to analyze performance
    - Evaluation System: Comprehensive evaluation tools to validate agentic workflows

The package includes custom functions for:
    - get_menu: Retrieve the complete flower shop menu
    - get_price: Get pricing information for specific items
    - add_to_cart: Add items to the shopping cart
    - remove_from_cart: Remove items from the shopping cart
    - clear_cart: Clear all items from the cart
    - view_cart: Display current cart contents

Example:
    To use this package, install it in development mode and run with NAT:

    ```bash
    uv pip install -e .
    nat start console --config_file configs/config.yml --input "Show me the menu"
    ```
"""

__version__ = "0.1.0"
