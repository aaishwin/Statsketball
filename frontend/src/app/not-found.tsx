import Link from "next/link";
import { ArrowLeftIcon } from "@/lib/icons";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center flex-1 px-5 pt-28 pb-32 text-center">
      <p className="text-[14px] font-semibold uppercase tracking-[0.12em] text-ink-muted mb-4">
        404
      </p>
      <h1 className="text-[clamp(2rem,4vw,3.2rem)] font-bold leading-[1.05] tracking-[-0.025em] text-balance mb-3">
        Page not found
      </h1>
      <p className="max-w-[420px] text-[15px] leading-relaxed text-ink-muted mb-8">
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
        Head back to search and find player comparisons instead.
      </p>
      <Link href="/" className="btn-subtle inline-flex">
        <ArrowLeftIcon size={14} />
        Back to search
      </Link>
    </div>
  );
}
