"use client";

/*
 * PlayerSearchInput — compact glass typeahead for the chart surfaces.
 * Mirrors the home-page search behavior (200ms debounce, arrow-key
 * navigation, outside-click dismiss) in a smaller footprint.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SearchIcon } from "@/lib/icons";
import { suggestPlayers } from "@/lib/api";
import type { PlayerSuggestion } from "@/lib/types";

interface PlayerSearchInputProps {
  placeholder: string;
  onSelect: (player: { entity_id: string; entity_name: string }) => void;
}

export default function PlayerSearchInput({
  placeholder,
  onSelect,
}: PlayerSearchInputProps) {
  const [value, setValue] = useState("");
  const [debounced, setDebounced] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(value.trim()), 200);
    return () => clearTimeout(t);
  }, [value]);

  const { data: suggestions } = useQuery<PlayerSuggestion[]>({
    queryKey: ["suggest", debounced],
    queryFn: () => suggestPlayers(debounced, 6),
    enabled: debounced.length >= 2 && open,
    staleTime: 30_000,
  });

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const select = useCallback(
    (s: PlayerSuggestion) => {
      setValue(s.entity_name);
      setOpen(false);
      setActive(0);
      onSelect({ entity_id: s.entity_id, entity_name: s.entity_name });
    },
    [onSelect]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open || !suggestions?.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((p) => (p < suggestions.length - 1 ? p + 1 : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((p) => (p > 0 ? p - 1 : suggestions.length - 1));
    } else if (e.key === "Enter" && suggestions[active]) {
      e.preventDefault();
      select(suggestions[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div ref={containerRef} className="relative w-full max-w-85">
      <div className="relative">
        <SearchIcon
          size={15}
          className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-muted pointer-events-none"
        />
        <input
          type="text"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setOpen(true);
            setActive(0);
          }}
          onFocus={() => value.trim().length >= 2 && setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          role="combobox"
          aria-expanded={open && !!suggestions?.length}
          aria-controls="player-search-listbox"
          aria-autocomplete="list"
          aria-label={placeholder}
          className="input-glass w-full pl-9 pr-3 py-2.5 text-[14px]"
        />
      </div>

      {open && suggestions && suggestions.length > 0 && (
        <ul
          id="player-search-listbox"
          role="listbox"
          className="absolute z-100 mt-2 w-full glass-inner-heavy rounded-xl overflow-hidden py-1"
        >
          {suggestions.map((s, i) => (
            <li key={s.entity_id} role="option" aria-selected={i === active}>
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  select(s);
                }}
                onMouseEnter={() => setActive(i)}
                className={`w-full text-left px-3.5 py-2.5 text-[14px] flex items-baseline gap-2 transition-colors duration-150 ${
                  i === active ? "bg-[oklch(1_0_0/0.06)]" : ""
                }`}
              >
                <span className="text-ink">{s.entity_name}</span>
                <span className="font-mono text-[11px] text-ink-muted">
                  {String(s.metadata?.position ?? s.metadata?.primary_position ?? "")}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
