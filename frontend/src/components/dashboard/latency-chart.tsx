"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Legend } from "recharts"
import { performanceMetrics } from "@/lib/mock-data"

export function LatencyChart() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Latency Over Time</CardTitle>
        <CardDescription>Average response time in milliseconds</CardDescription>
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
              dataKey="latency"
              stroke="#8884d8"
              strokeWidth={2}
              dot={false}
              name="Latency (ms)"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
