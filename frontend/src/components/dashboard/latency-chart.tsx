"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "recharts"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { performanceMetrics } from "@/lib/mock-data"

export function LatencyChart() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Latency Over Time</CardTitle>
        <CardDescription>Average response time in milliseconds</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            latency: {
              label: "Latency",
              color: "hsl(var(--chart-2))",
            },
          }}
          className="h-[200px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="var(--color-latency)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
