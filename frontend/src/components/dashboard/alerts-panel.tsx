"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { recentAlerts } from "@/lib/mock-data"
import { AlertCircle, CheckCircle, Info } from "lucide-react"

const alertIcons = {
  warning: AlertCircle,
  info: Info,
  success: CheckCircle,
}

const alertColors = {
  warning: "text-yellow-500",
  info: "text-blue-500",
  success: "text-green-500",
}

export function AlertsPanel() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Alerts</CardTitle>
        <CardDescription>System notifications and warnings</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {recentAlerts.map((alert) => {
            const Icon = alertIcons[alert.type]
            const colorClass = alertColors[alert.type]
            
            return (
              <div key={alert.id} className="flex items-start gap-3">
                <Icon className={`h-4 w-4 mt-0.5 ${colorClass}`} />
                <div className="flex-1 space-y-1">
                  <p className="text-sm">{alert.message}</p>
                  <p className="text-xs text-muted-foreground">{alert.time}</p>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
