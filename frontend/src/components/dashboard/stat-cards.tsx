"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Database, Gauge, Activity, HardDrive } from "lucide-react"
import { systemHealth } from "@/lib/mock-data"

const stats = [
  {
    label: "Total Vectors",
    value: systemHealth.totalVectors.toLocaleString(),
    icon: Database,
    change: "+12.4K today",
  },
  {
    label: "Throughput",
    value: "2,340 ops/s",
    icon: Gauge,
    change: "Peak: 2,500",
  },
  {
    label: "Avg Latency",
    value: "1.3 ms",
    icon: Activity,
    change: "-0.2ms vs yesterday",
  },
  {
    label: "Memory Usage",
    value: `${systemHealth.memoryUsed} / ${systemHealth.memoryTotal} GB`,
    icon: HardDrive,
    change: `${Math.round((systemHealth.memoryUsed / systemHealth.memoryTotal) * 100)}% utilized`,
  },
]

export function StatCards() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.label}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">{stat.label}</span>
                <span className="text-xl font-semibold font-mono">{stat.value}</span>
                <span className="text-xs text-muted-foreground">{stat.change}</span>
              </div>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <stat.icon className="h-5 w-5 text-primary" />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
