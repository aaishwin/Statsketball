"use client";

import { useEffect, useRef } from "react";

/*
 * Particle Background — animated vector-space field (DESIGN.md Amendment A)
 *
 * A full-screen 2D canvas rendering ~80-120 dots drifting in Brownian
 * motion. Each dot is a visual metaphor for a player vector in embedding
 * space. The background IS the product's core concept made ambient.
 *
 * Dots render in --ink-secondary at 8-12% opacity. When the mouse moves,
 * dots within a 120px radius brighten and shift toward the cursor.
 *
 * Performance: requestAnimationFrame, pauses on tab hide.
 * Accessibility: prefers-reduced-motion gets static dots, no drift.
 * No WebGL, no dependencies.
 */

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseOpacity: number;
  size: number;
}

const PARTICLE_COUNT = 100;
const MOUSE_RADIUS = 120;
const DRIFT_SPEED = 0.15;

export function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const rafRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let width = 0;
    let height = 0;
    let dpr = 1;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.scale(dpr, dpr);
    };

    const initParticles = () => {
      const particles: Particle[] = [];
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * DRIFT_SPEED,
          vy: (Math.random() - 0.5) * DRIFT_SPEED,
          baseOpacity: 0.06 + Math.random() * 0.06,
          size: Math.random() < 0.3 ? 2 : 1,
        });
      }
      particlesRef.current = particles;
    };

    resize();
    initParticles();

    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseLeave = () => {
      mouseRef.current = { x: -1000, y: -1000 };
    };

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseout", onMouseLeave);

    let isHidden = false;
    const onVisibility = () => {
      isHidden = document.hidden;
      if (!isHidden && !reducedMotion) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    const animate = () => {
      if (isHidden) return;

      ctx.clearRect(0, 0, width, height);

      const mouse = mouseRef.current;
      const particles = particlesRef.current;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        if (!reducedMotion) {
          // Brownian motion — add tiny random velocity changes
          p.vx += (Math.random() - 0.5) * 0.02;
          p.vy += (Math.random() - 0.5) * 0.02;
          // Clamp velocity
          p.vx = Math.max(-DRIFT_SPEED, Math.min(DRIFT_SPEED, p.vx));
          p.vy = Math.max(-DRIFT_SPEED, Math.min(DRIFT_SPEED, p.vy));
          // Update position
          p.x += p.vx;
          p.y += p.vy;
          // Wrap around edges
          if (p.x < -10) p.x = width + 10;
          if (p.x > width + 10) p.x = -10;
          if (p.y < -10) p.y = height + 10;
          if (p.y > height + 10) p.y = -10;
        }

        // Mouse interaction — brighten and attract
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        let opacity = p.baseOpacity;
        let drawX = p.x;
        let drawY = p.y;

        if (dist < MOUSE_RADIUS) {
          const influence = 1 - dist / MOUSE_RADIUS;
          opacity = p.baseOpacity + influence * 0.15;
          // Slight magnetic pull toward cursor
          if (!reducedMotion) {
            drawX = p.x + dx * influence * 0.03;
            drawY = p.y + dy * influence * 0.03;
          }
        }

        // Draw the dot — warm grey, no hue
        ctx.beginPath();
        ctx.arc(drawX, drawY, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(194, 194, 191, ${opacity})`;
        ctx.fill();
      }

      if (!reducedMotion) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseout", onMouseLeave);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="particle-canvas"
        aria-hidden="true"
      />
      <div className="vignette-layer" aria-hidden="true" />
    </>
  );
}
