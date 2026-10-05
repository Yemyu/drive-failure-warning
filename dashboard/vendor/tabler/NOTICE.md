# Tabler 使用说明

## 当前资产（2026-10-03）

- 上游项目：https://github.com/tabler/tabler
- 固定版本：**@tabler/core 1.0.0**，MIT License。
- `tabler.min.css` 与 `LICENSE` 从官方 npm 包直接提取，未修改；下载地址、字节数及 SHA-256 见 `ASSETS.json`。
- `dashboard/template_signal.html` 使用这份样式表的侧栏、导航、卡片、按钮、表单与表格组件；项目专用布局和图表样式写在模板中。
- 未引入 Tabler JavaScript、第三方插件、照片或付费模块。导航、筛选和案例回放由项目自身 JavaScript 实现。
- 样式表无 `@import`、无 `@font-face`，35 个 `url()` 均为内联 SVG 数据；没有字体、图片或 CDN 请求。默认字体栈包含 Inter，但仅使用本机已有字体；页面可覆盖为系统字体。
- GitHub Pages 从同一站点加载 `../vendor/tabler/tabler.min.css`，不依赖外部 CDN。
