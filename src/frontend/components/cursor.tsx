"use client";

import { useEffect, useRef, useState } from "react";

/*
 * Custom Cursor — dot + ring (DESIGN.md Amendment B)
 *
 * Dot: 6px solid, tracks near-instantly (lerp 0.85).
 * Ring: 28px border, follows with spring lag (lerp 0.15).
 *
 * States:
 *   default       — dot + ring visible
 *   hover         — ring expands to 40px, dot hides (links, buttons, chips)
 *   selectable    — ring expands to 56px, amber tint (result rows, player cards)
 *
 * Desktop only (pointer: fine). Touch returns null.
 * Reduced motion: ring lag disabled, both track instantly.
 * mix-blend-mode: difference ensures visibility over any surface.
 */

type CursorState = "default" | "hover" | "selectable";

const INTERACTIVE_SELECTOR = "a, button, input, [data-cursor='hover']";
const SELECTABLE_SELECTOR = "[data-cursor='selectable']";

export function Cursor() {
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<CursorState>("default");
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    // Only enable on fine pointer (desktop with mouse). Touch keeps native.
    if (!window.matchMedia("(pointer: fine)").matches) return;

    /* The CSS `cursor: none` rules are gated behind `html.custom-cursor`,
       set only while this component is mounted. No JS (or failed hydration)
       means the native cursor stays visible — never a cursorless page. */
    document.documentElement.classList.add("custom-cursor");
    // Mount-gated: this effect runs once on the client, after hydration.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEnabled(true);

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dotLerp = 0.85;
    const ringLerp = reducedMotion ? 1.0 : 0.15;

    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let dotX = mouseX;
    let dotY = mouseY;
    let ringX = mouseX;
    let ringY = mouseY;
    let rafId = 0;

    const onMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;

      // Determine cursor state based on element under pointer
      const el = e.target as HTMLElement;
      if (el?.closest(SELECTABLE_SELECTOR)) {
        setState("selectable");
      } else if (el?.closest(INTERACTIVE_SELECTOR)) {
        setState("hover");
      } else {
        setState("default");
      }
    };

    const tick = () => {
      dotX += (mouseX - dotX) * dotLerp;
      dotY += (mouseY - dotY) * dotLerp;
      ringX += (mouseX - ringX) * ringLerp;
      ringY += (mouseY - ringY) * ringLerp;

      if (dotRef.current) {
        dotRef.current.style.transform = `translate(${dotX}px, ${dotY}px) translate(-50%, -50%)`;
      }
      if (ringRef.current) {
        ringRef.current.style.transform = `translate(${ringX}px, ${ringY}px) translate(-50%, -50%)`;
      }
      rafId = requestAnimationFrame(tick);
    };

    window.addEventListener("mousemove", onMove);
    rafId = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(rafId);
      document.documentElement.classList.remove("custom-cursor");
    };
  }, []);

  if (!enabled) return null;

  const ringClass =
    state === "selectable"
      ? "cursor-ring cursor-ring-selectable"
      : state === "hover"
        ? "cursor-ring cursor-ring-hover"
        : "cursor-ring";

  const dotClass = state === "hover" ? "cursor-dot cursor-dot-hidden" : "cursor-dot";

  return (
    <>
      <div ref={ringRef} className={ringClass} aria-hidden="true" />
      <div ref={dotRef} className={dotClass} aria-hidden="true" />
    </>
  );
}
