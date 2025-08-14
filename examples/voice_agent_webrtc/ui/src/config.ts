// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD 2-Clause License

export const RTC_CONFIG = {};

const host = window.location.hostname;

export const RTC_OFFER_URL = `ws://${host}:7860/ws`;
export const POLL_PROMPT_URL = `http://${host}:7860/get_prompt`;

// Set to true to use dynamic prompt mode, false for default mode
export const DYNAMIC_PROMPT = false;
