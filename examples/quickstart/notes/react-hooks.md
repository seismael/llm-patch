# React Hooks — quick reference

`useState`, `useEffect`, `useMemo`, `useCallback`, `useRef`, `useReducer`.

## `useState`

```tsx
const [count, setCount] = useState(0);
```

## `useEffect` cleanup

```tsx
useEffect(() => {
  const id = setInterval(tick, 1000);
  return () => clearInterval(id);
}, []);
```

## `useMemo` vs `useCallback`

- `useMemo(fn, deps)` memoizes a **value**.
- `useCallback(fn, deps)` memoizes a **function reference**.

Always wrap dependencies that are themselves functions in
`useCallback` to keep child components stable.
