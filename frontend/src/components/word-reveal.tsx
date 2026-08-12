"use client";

import { motion } from "motion/react";

/*
 * WordReveal — word-level opacity + y reveal (Amendment C)
 *
 * Replaces LetterReveal (per-character blur) which caused GPU flicker.
 * Splits text into words, staggers opacity + y. No filter: blur() on
 * individual elements. The parent container may have a single blur.
 *
 * Easing: expo out (0.16, 1, 0.3, 1) for a confident settle.
 */

interface WordRevealProps {
  text: string;
  delay?: number;
  stagger?: number;
  className?: string;
}

export function WordReveal({
  text,
  delay = 0,
  stagger = 0.06,
  className,
}: WordRevealProps) {
  const words = text.split(" ");

  return (
    <motion.span
      className={className}
      aria-label={text}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: {
          transition: {
            staggerChildren: stagger,
            delayChildren: delay,
          },
        },
      }}
    >
      {words.map((word, i) => (
        <span key={i} style={{ display: "inline-block", overflow: "hidden" }}>
          <motion.span
            aria-hidden="true"
            style={{ display: "inline-block" }}
            variants={{
              hidden: { opacity: 0, y: "100%" },
              visible: {
                opacity: 1,
                y: "0%",
                transition: {
                  duration: 0.6,
                  ease: [0.16, 1, 0.3, 1],
                },
              },
            }}
          >
            {word}
            {i < words.length - 1 ? "\u00A0" : ""}
          </motion.span>
        </span>
      ))}
    </motion.span>
  );
}
