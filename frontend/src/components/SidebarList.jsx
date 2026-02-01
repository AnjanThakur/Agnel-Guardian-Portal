import React from 'react'
import { FileText, CheckCircle2, Loader2, Circle, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

export function SidebarList({ files, activeIndex, onSelect, processingIndex, results }) {

    // files is an array of File objects
    // results is a Map or Object: { index: { status: 'pending'|'loading'|'done'|'error', data: ... } }

    return (
        <Card className="border-none shadow-xl shadow-slate-200/40 bg-white/50 backdrop-blur-sm h-full max-h-[600px] flex flex-col">
            <CardHeader className="pb-3 border-b border-border/40">
                <CardTitle className="text-lg font-serif flex items-center justify-between">
                    <span>Inbox</span>
                    <span className="text-xs font-sans font-medium text-muted-foreground bg-slate-100 px-2 py-1 rounded-full">
                        {files.length}
                    </span>
                </CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1 overflow-y-auto scrollbar-thin">
                <div className="flex flex-col">
                    {files.map((file, idx) => {
                        const isActive = idx === activeIndex
                        const result = results[idx] || {}
                        const status = result.status || 'pending' // pending, loading, success, error

                        return (
                            <button
                                key={`${file.name}-${idx}`}
                                onClick={() => onSelect(idx)}
                                className={cn(
                                    "flex items-center gap-3 p-4 text-left transition-all border-b border-border/40 hover:bg-slate-50",
                                    isActive ? "bg-primary/5 border-l-4 border-l-primary" : "border-l-4 border-l-transparent"
                                )}
                            >
                                {/* Icon based on Status */}
                                <div className="shrink-0">
                                    {status === 'loading' && <Loader2 className="w-4 h-4 text-accent animate-spin" />}
                                    {status === 'success' && <CheckCircle2 className="w-4 h-4 text-green-600" />}
                                    {status === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                                    {status === 'pending' && <Circle className="w-4 h-4 text-slate-300" />}
                                </div>

                                <div className="min-w-0 flex-1">
                                    <p className={cn("text-sm font-medium truncate", isActive ? "text-primary" : "text-foreground")}>
                                        {file.name}
                                    </p>
                                    <p className="text-xs text-muted-foreground truncate">
                                        {(file.size / 1024).toFixed(0)} KB • {status.charAt(0).toUpperCase() + status.slice(1)}
                                    </p>
                                </div>
                            </button>
                        )
                    })}

                    {files.length === 0 && (
                        <div className="p-8 text-center text-muted-foreground text-sm italic">
                            No files uploaded yet.
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
