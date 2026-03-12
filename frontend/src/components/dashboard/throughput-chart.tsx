"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "recharts"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { performanceMetrics } from "@/lib/mock-data"

export function ThroughputChart() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Throughput Over Time</CardTitle>
        <CardDescription>Operations per second</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            throughput: {
              label: "Throughput",
              color: "hsl(var(--chart-1))",
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
                dataKey="throughput"
                stroke="var(--color-throughput)"
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
