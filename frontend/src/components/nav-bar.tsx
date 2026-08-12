"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "motion/react";
import { getIndexInfo } from "@/lib/api";

/*
 * Fluid Island Nav — CSS-only two-layer text slide (Unseen-style).
 *
 * Each nav link has overflow:hidden with two stacked text layers.
 * On hover (or when active), the default layer slides up and out
 * while the hover layer slides up into view. Pure CSS transition,
 * no Framer Motion per-character — eliminates the jitter caused by
 * animate/whileHover conflicts in the previous implementation.
 *
 * Live counter: mono tabular number showing index size.
 */

const NAV_LINKS = [
  { href: "/", label: "Search", num: "01" },
  { href: "/archetypes", label: "Archetypes", num: "02" },
];

export function NavBar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [vectorCount, setVectorCount] = useState<number | null>(null);

  /* ── Scroll detection ── */
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  /* ── Fetch index count for live counter ── */
  useEffect(() => {
    getIndexInfo("player")
      .then((d) => setVectorCount(d.vector_count))
      .catch(() => {});
  }, []);

  /* ── Lock body scroll when menu open ── */
  useEffect(() => {
    document.body.classList.toggle("body-locked", menuOpen);
    return () => document.body.classList.remove("body-locked");
  }, [menuOpen]);

  const closeMenu = useCallback(() => setMenuOpen(false), []);

  return (
    <>
      {/* ═══ Fluid Island Pill ═══ */}
      <header
        className={`nav-island ${scrolled ? "nav-island-scrolled" : ""}`}
        role="banner"
      >
        {/* Brand wordmark + amber dot */}
        <Link href="/" className="nav-brand" onClick={closeMenu}>
          StatsketBall
          <span className="nav-brand-dot" aria-hidden="true" />
        </Link>

        {/* Desktop nav links — CSS-only two-layer slide */}
        <nav className="hidden sm:flex items-center gap-6" aria-label="Main navigation">
          {NAV_LINKS.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`nav-link ${active ? "nav-link-active" : "nav-link-inactive"}`}
              >
                <span className="nav-link-clip">
                  <span className="nav-link-text-default">{label}</span>
                  <span className="nav-link-text-hover">{label}</span>
                </span>
              </Link>
            );
          })}
        </nav>


        {/* Hamburger — mobile only. Morphs into X on open. */}
        <button
          type="button"
          className={`sm:hidden flex flex-col gap-[4.5px] p-1 -mr-1 ${
            menuOpen ? "hamburger-open" : ""
          }`}
          onClick={() => setMenuOpen((prev) => !prev)}
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
        >
          <span className="hamburger-line" />
          <span className="hamburger-line" />
          <span className="hamburger-line" />
        </button>
      </header>

      {/* ═══ Mobile Full-Screen Menu Overlay ═══ */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            className="menu-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          >
            {NAV_LINKS.map(({ href, label, num }, i) => (
              <motion.div
                key={href}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
                transition={{
                  duration: 0.50,
                  ease: [0.22, 1, 0.36, 1],
                  delay: 0.08 + i * 0.08,
                }}
                className="flex items-baseline gap-4"
              >
                <span className="font-mono text-[13px] tabular-nums text-ink-disabled">
                  {num}
                </span>
                <Link
                  href={href}
                  className="menu-link menu-link-visible"
                  onClick={closeMenu}
                  style={{
                    color: pathname === href
                      ? "oklch(0.65 0.20 40)"
                      : undefined,
                  }}
                >
                  {label}
                </Link>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
