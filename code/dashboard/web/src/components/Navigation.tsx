'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

export type DashboardTab = {
  id: string;
  label: string;
  href: string;
  shortcut: string;
};

export const DASHBOARD_TABS: DashboardTab[] = [
  { id: 'overview', label: 'Overview', href: '/', shortcut: '1' },
  { id: 'compare', label: 'Compare', href: '/compare', shortcut: '2' },
  { id: 'profiler', label: 'Profiler', href: '/profiler', shortcut: '3' },
  { id: 'memory', label: 'Memory', href: '/memory', shortcut: '4' },
  { id: 'multi-gpu', label: 'Multi-GPU', href: '/multi-gpu', shortcut: '5' },
  { id: 'history', label: 'History', href: '/history', shortcut: '6' },
  { id: 'system', label: 'System', href: '/system', shortcut: '7' },
  { id: 'cluster', label: 'Cluster', href: '/cluster', shortcut: '8' },
];

export function Navigation() {
  const pathname = usePathname();
  const current = pathname === '/' ? '/' : pathname;

  return (
    <nav className="flex flex-wrap items-center gap-2">
      {DASHBOARD_TABS.map((tab) => {
        const isActive = current === tab.href;
        return (
          <Link
            key={tab.id}
            href={tab.href}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-full border text-sm transition-all',
              isActive
                ? 'bg-white/10 border-white/20 text-white shadow-sm'
                : 'bg-white/5 border-white/10 text-white/60 hover:text-white hover:border-white/20'
            )}
          >
            <span>{tab.label}</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/10 text-white/50">
              {tab.shortcut}
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
