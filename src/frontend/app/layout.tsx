import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { Providers } from "@/lib/providers";
import { NavBar } from "@/components/nav-bar";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Statsketball: NBA Player Comparisons",
  description: "Find stylistically similar NBA players using ML-powered vector search.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Statsketball, NBA Player Comparisons",
    description: "Find stylistically similar NBA players using ML-powered vector search.",
    siteName: "Statsketball",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0a0a",
  colorScheme: "dark",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={cn("h-full", "antialiased", GeistSans.variable, GeistMono.variable, "font-sans")}>
      <body className="min-h-full flex flex-col relative">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-[600] focus:px-4 focus:py-2 focus:bg-primary focus:text-ink focus:rounded-full focus:outline-none"
        >
          Skip to content
        </a>
        <Providers>
          <NavBar />
          <main id="main-content" className="relative z-[1] flex-1 flex flex-col">{children}</main>
        </Providers>
      </body>
    </html>
  );
}

