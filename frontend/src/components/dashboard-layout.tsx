import * as React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface DashboardLayoutProps {
  title: string
  description: string
  actions?: React.ReactNode
  children: React.ReactNode
}

export function DashboardLayout({ title, description, actions, children }: DashboardLayoutProps) {
  return (
    <div className="flex min-h-screen w-full flex-col">
      <header className="sticky top-0 z-50 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6">
        <nav className="hidden flex-col gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6">
          <a href="#" className="flex items-center gap-2 text-lg font-semibold md:text-base">
            <span className="font-bold">HybridVectorDB</span>
          </a>
          <a href="#" className="text-foreground transition-colors hover:text-foreground/80">
            Dashboard
          </a>
          <a href="#" className="text-muted-foreground transition-colors hover:text-foreground">
            Vectors
          </a>
          <a href="#" className="text-muted-foreground transition-colors hover:text-foreground">
            Benchmarks
          </a>
          <a href="#" className="text-muted-foreground transition-colors hover:text-foreground">
            Configuration
          </a>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          {actions}
        </div>
      </header>
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="mx-auto grid w-full max-w-6xl gap-2">
          <h1 className="text-3xl font-semibold">{title}</h1>
          <p className="text-muted-foreground">{description}</p>
        </div>
        <div className="mx-auto w-full max-w-6xl">
          {children}
        </div>
      </main>
    </div>
  )
}
