"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Legend } from "recharts"
import { performanceMetrics } from "@/lib/mock-data"

export function ThroughputChart() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Throughput Over Time</CardTitle>
        <CardDescription>Operations per second</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={performanceMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="throughput"
              stroke="#82ca9d"
              strokeWidth={2}
              dot={false}
              name="Throughput (ops/s)"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
