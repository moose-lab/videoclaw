"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Film, FolderOpen, Workflow, Sparkles } from "lucide-react";

const links = [
  { href: "/", label: "Dashboard", icon: FolderOpen },
  { href: "/generate", label: "Generate", icon: Sparkles },
  { href: "/flow", label: "ClawFlow", icon: Workflow },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="border-b bg-background">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4">
        <Link href="/" className="flex items-center gap-2 font-bold text-lg">
          <Film className="h-5 w-5 text-primary" />
          <span>VideoClaw</span>
        </Link>
        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
                  active
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
