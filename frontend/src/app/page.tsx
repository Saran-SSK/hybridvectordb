"use client"

import { DashboardLayout } from "@/components/dashboard-layout"
import { StatCards } from "@/components/dashboard/stat-cards"
import { ThroughputChart } from "@/components/dashboard/throughput-chart"
import { LatencyChart } from "@/components/dashboard/latency-chart"
import { AlertsPanel } from "@/components/dashboard/alerts-panel"
import { RoutingCard } from "@/components/dashboard/routing-card"
import { Badge } from "@/components/ui/badge"

export default function DashboardPage() {
  return (
    <DashboardLayout
      title="HybridVectorDB Console"
      description="Real-time monitoring and metrics"
      actions={
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono text-xs">
            Last updated: just now
          </Badge>
          <Badge className="border-0 bg-success/20 text-success">
            All Systems Operational
          </Badge>
        </div>
      }
    >
      <div className="flex flex-col gap-4 md:gap-6">
        <StatCards />
        <div className="grid gap-4 md:gap-6 lg:grid-cols-3">
          <ThroughputChart />
          <LatencyChart />
        </div>
        <div className="grid gap-4 md:gap-6 lg:grid-cols-2">
          <RoutingCard />
          <AlertsPanel />
        </div>
      </div>
    </DashboardLayout>
  )
}
