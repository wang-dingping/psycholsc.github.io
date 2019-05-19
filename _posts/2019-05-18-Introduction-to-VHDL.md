---
layout: post
comments: true
title:  "Introduction to VHDL"
excerpt: "-"
date:   2019-05-18 12:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---

[TOC]

## VHDL

最简单的`VHDL`程序至少由以下部分组成

- `Library`。一般就是常用的`IEEE`标准库

```VHDL
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
```

- `Package`
- `Entity`
- `Architecture`
- `Configuration`

