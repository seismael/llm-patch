# Tailwind tips

- Prefer `className="flex items-center gap-2"` over hand-rolled CSS.
- Use the `@apply` directive in component CSS only for repeated utility
  combinations; otherwise inline.
- Dark mode: `dark:` variant + the `class` strategy in
  `tailwind.config.js`.

## Common patterns

| Need | Utilities |
|---|---|
| Center a column | `flex flex-col items-center` |
| Equal-height grid | `grid grid-cols-3 gap-4` |
| Truncate text | `truncate` (single line) or `line-clamp-3` (multi) |
| Responsive padding | `p-4 md:p-8 lg:p-12` |
