"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { routingDecisions } from "@/lib/mock-data"
import { Cpu, Zap } from "lucide-react"

export function RoutingCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Hybrid Routing</CardTitle>
        <CardDescription>CPU vs GPU query distribution</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {routingDecisions.map((decision) => (
            <div key={decision.time} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">{decision.time}</span>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-blue-500" />
                  <span className="text-sm">{decision.cpu.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-500" />
                  <span className="text-sm">{decision.gpu.toLocaleString()}</span>
                </div>
                <div className="text-sm font-mono text-muted-foreground">
                  {decision.total.toLocaleString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
