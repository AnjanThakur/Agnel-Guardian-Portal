import React from 'react'
import { Lightbulb, AlertTriangle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Card, CardContent } from './ui/card'

export function AISummaryReport({ data }) {
    if (!data) return null

    // Error State
    if (data.error) {
        return (
            <Card className="border-red-200 bg-red-50/50 mt-6">
                <CardContent className="p-6">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-red-100">
                            <AlertTriangle className="w-5 h-5 text-red-600" />
                        </div>
                        <div>
                            <p className="font-semibold text-red-900 font-serif">Analysis Failed</p>
                            <p className="text-sm text-red-700">{data.details || data.error}</p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        )
    }

    // Sentiment icon
    const SentimentIcon = data.sentiment_overview === 'Positive' ? TrendingUp
        : data.sentiment_overview === 'Negative' ? TrendingDown
            : Minus

    // Sentiment Colors - refined for premium look (subtle backgrounds)
    const sentimentColor = data.sentiment_overview === 'Positive' ? 'text-green-700 bg-green-50 border-green-200'
        : data.sentiment_overview === 'Negative' ? 'text-red-700 bg-red-50 border-red-200'
            : 'text-amber-700 bg-amber-50 border-amber-200'

    return (
        <Card className="border-primary/10 bg-white shadow-lg shadow-purple-900/5 mt-6 overflow-hidden">
            <div className="h-1 w-full bg-gradient-to-r from-primary/40 via-purple-400 to-primary/40 opacity-50"></div>
            <CardContent className="p-6">
                {/* Header: Just Sentiment Badge on right */}
                <div className="flex items-center justify-end mb-4">
                    <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${sentimentColor}`}>
                        <SentimentIcon className="w-3.5 h-3.5" />
                        <span className="text-xs font-semibold uppercase tracking-wider">{data.sentiment_overview}</span>
                    </div>
                </div>

                {/* Executive Summary */}
                <div className="mb-6">
                    <p className="text-base text-slate-700 leading-relaxed font-serif">
                        {data.executive_summary}
                    </p>
                </div>

                {/* Recommendations */}
                {data.actionable_insights && data.actionable_insights.length > 0 && (
                    <div className="pt-6 border-t border-slate-100">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="p-1.5 bg-amber-50 rounded-md">
                                <Lightbulb className="w-4 h-4 text-amber-500" />
                            </div>
                            <h4 className="font-semibold text-sm text-slate-800 font-serif tracking-tight">Key Recommendations</h4>
                        </div>
                        <div className="grid gap-3">
                            {data.actionable_insights.map((insight, idx) => (
                                <div key={idx} className="group flex items-start gap-3 text-sm text-slate-600 hover:text-slate-900 transition-colors p-3 rounded-lg bg-slate-50 border border-slate-100/50 hover:bg-white hover:shadow-sm hover:border-slate-200">
                                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-white border border-slate-200 text-slate-500 text-[10px] flex items-center justify-center font-bold shadow-sm group-hover:border-primary/30 group-hover:text-primary transition-colors">
                                        {idx + 1}
                                    </span>
                                    <span className="leading-relaxed">{insight}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
