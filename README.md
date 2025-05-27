# TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression

## Overview

**TLDR** is an efficient training and inference framework for Large Language Models (LLMs) that compresses reasoning chains without sacrificing accuracy. By dynamically re-weighting short and long chain-of-thought (CoT) data during training, our method eliminates redundant reasoning steps, yielding concise outputs with comparable or superior accuracy.

🚀 Dynamic Ratio Training: No need for complex annotations or multi-model interpolation.

⚡ Efficiency: Reduces output token length by up to 40% while maintaining reasoning accuracy.

📊 Versatile: Validated on DeepSeek-R1-Distill-7B/14B and multiple mathematical reasoning benchmarks (GSM8K, MATH500, AIME, etc.).


## Highlights

Dynamic Re-weighting: Automatically balances System-1 (concise/intuitive) and System-2 (detailed/deliberative) reasoning samples.

Plug-and-Play: Can be applied to any LLM reasoning task with minimal adaptation.

No Expensive Annotations: Avoids tedious data labeling and parameter search.
