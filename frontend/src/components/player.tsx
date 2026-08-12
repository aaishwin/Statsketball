/*
 * Player identity primitives — shared across search, archetypes, compare.
 *
 * DESIGN.md Imagery: a missing photo is "monospace initials on
 * --surface-elevated, styled like a scoreboard nameplate."
 *
 * Uses Radix Avatar primitives for the image/fallback state machine
 * (accessibility + graceful degradation when headshots aren't available).
 * Visual style follows the design-system avatar pattern with a subtle
 * inner border overlay — see avatar.tsx for the canonical implementation.
 */

import { Avatar as AvatarPrimitive } from "radix-ui";
import { cn } from "@/lib/utils";
import { useHeadshotUrl } from "@/lib/headshots";

export function initials(name: string): string {
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2);
}

export function PlayerAvatar({
  name,
  size = "md",
  highlight = false,
  className,
}: {
  name: string;
  /** Basketball-Reference entity slug (e.g. "jamesle01"). Retained for
   *  future use but headshot lookup is now name-based. */
  entityId?: string;
  size?: "sm" | "md" | "lg";
  highlight?: boolean;
  className?: string;
}) {
  const headshotUrl = useHeadshotUrl(name);
  const imageSrc = headshotUrl;

  return (
    <AvatarPrimitive.Root
      data-slot="avatar"
      data-size={size}
      className={cn(
        "avatar-circle group/avatar relative flex shrink-0 rounded-full select-none",
        "after:absolute after:inset-0 after:rounded-full after:border after:border-border after:mix-blend-lighten",
        size === "sm" && "size-8 text-[11px]",
        size === "md" && "size-10 text-[13px]",
        size === "lg" && "size-12 text-[14px]",
        highlight
          ? "bg-primary text-[oklch(0.10_0.005_40)]"
          : "bg-surface-raised text-ink-soft",
        className,
      )}
    >
      {imageSrc && (
        <AvatarPrimitive.Image
          data-slot="avatar-image"
          src={imageSrc}
          alt={name}
          className="aspect-square size-full rounded-full object-cover"
        />
      )}
      <AvatarPrimitive.Fallback
        data-slot="avatar-fallback"
        className={cn(
          "flex size-full items-center justify-center rounded-full",
          "font-bold tracking-[0.02em] uppercase",
        )}
      >
        {initials(name)}
      </AvatarPrimitive.Fallback>
    </AvatarPrimitive.Root>
  );
}

export function ScorePill({
  score,
  highlight = false,
  className,
}: {
  /** 0..1 similarity score */
  score: number;
  highlight?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "score-pill",
        highlight
          ? "bg-primary-ambient text-primary"
          : "bg-surface-raised text-ink-soft",
        className,
      )}
    >
      {(score * 100).toFixed(0)}%
    </span>
  );
}

export function RankBadge({
  rank,
  highlight = false,
  className,
}: {
  rank: number;
  highlight?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "rank-badge",
        highlight ? "text-primary" : "text-ink-muted",
        className,
      )}
    >
      {String(rank).padStart(2, "0")}
    </span>
  );
}
