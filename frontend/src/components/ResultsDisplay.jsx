import React, { useState } from 'react'
// import { Card, CardContent, CardHeader, CardTitle } from './ui/card' // We might use custom layout
import { CheckCircle2, ChevronRight, Code, FileText, Image as ImageIcon, Clock, BarChart3, MessageSquare } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'

// --- CONSTANTS ---
const LABELS = {
    "q1_teaching_learning_environment": "Teaching & Learning Environment",
    "q2_monitoring_students_progress": "Monitoring Student Progress",
    "q3_faculty_involvement": "Faculty Involvement",
    "q4_infrastructure_facilities": "Infrastructure & Facilities",
    "q5_learning_resources": "Learning Resources",
    "q6_study_environment_and_discipline": "Study Environment & Discipline",
    "q7_counselling_and_placements": "Counselling & Placements",
    "q8_support_facilities": "Support Facilities",
    "q9_parental_perception": "Parental Perception",
    "q10_holistic_development": "Holistic Development"
}

// --- SUB-COMPONENTS ---
function ScoreRow({ label, value, confidence }) {
    // Value is expected to be a number 1-4 or string "1"..."4"
    const valNum = parseInt(value)
    const isValid = !isNaN(valNum)

    // Progress Bar Width
    const percentage = isValid ? (valNum / 4) * 100 : 0

    // Color Logic
    let colorClass = "bg-primary"
    if (valNum >= 3) colorClass = "bg-green-600"
    else if (valNum === 2) colorClass = "bg-yellow-500"
    else if (valNum === 1) colorClass = "bg-red-500"

    return (
        <div className="flex items-center justify-between py-3 border-b border-border/40 last:border-0 group hover:bg-slate-50/50 px-2 rounded-lg transition-colors">
            <div className="flex-1 pr-4">
                <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors font-serif">
                    {label}
                </p>
                {confidence < 0.8 && (
                    <span className="text-[10px] text-amber-600 font-semibold bg-amber-50 px-1.5 py-0.5 rounded">
                        Low Confidence
                    </span>
                )}
            </div>

            <div className="flex items-center gap-4 w-40">
                {/* Progress Bar Track */}
                <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                        className={cn("h-full rounded-full transition-all duration-1000 ease-out", colorClass)}
                        style={{ width: `${percentage}%` }}
                    />
                </div>

                {/* Score Text */}
                <div className="w-8 text-right font-mono text-sm font-bold text-foreground">
                    {isValid ? valNum : '-'}
                    <span className="text-muted-foreground text-xs font-normal">/4</span>
                </div>
            </div>
        </div>
    )
}

function CommentSection({ text }) {
    if (!text) return null
    return (
        <div className="mt-6 pt-6 border-t border-border/60">
            <h4 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">
                <MessageSquare className="w-4 h-4" /> Comments
            </h4>
            <div className="bg-amber-50/50 border border-amber-100 p-4 rounded-xl text-sm font-serif leading-relaxed text-amber-900/80">
                "{text}"
            </div>
        </div>
    )
}


export function ResultsDisplay({ data, previewUrl, rawText }) {
    const [showRaw, setShowRaw] = useState(false)
    const [showJson, setShowJson] = useState(false)

    // Data Normalization
    // The backend returns { ratings: { q1: { value: 3, ... } }, comments: "..." } for new logic
    // OR just flat fields for old logic? 
    // Let's inspect 'data'.
    // Usually 'run_pta_free' returns { ratings: {...}, comments: "..." }

    const ratings = data.ratings || {}
    const comments = data.comments || ""
    const usage = data.usage || {}

    // If we don't have 'ratings' key, maybe it's the old flat structure?
    // We can try to use 'data' itself if 'ratings' is missing.
    const displayFields = Object.keys(ratings).length > 0 ? ratings : data

    const keys = Object.keys(displayFields)
    const sortedKeys = keys.sort((a, b) => {
        // Sort Q1..Q10 properly
        const getNum = (s) => parseInt(s.match(/q(\d+)/)?.[1] || 999)
        return getNum(a) - getNum(b)
    })

    // Check if we have mapped keys (PTA mode)
    const isPtaMode = keys.some(k => LABELS[k])

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">

            {/* 1. Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                {/* LEFT: Preview Image */}
                <div className="space-y-3">
                    <h3 className="text-lg font-serif font-semibold text-foreground flex items-center gap-2">
                        <ImageIcon className="w-4 h-4 text-accent" /> Document Source
                    </h3>
                    <div className="relative overflow-hidden rounded-xl border bg-white shadow-sm group">
                        {previewUrl ? (
                            <img
                                src={previewUrl}
                                alt="Document Preview"
                                className="w-full h-auto object-cover max-h-[500px] transition-transform duration-700 group-hover:scale-[1.02]"
                            />
                        ) : (
                            <div className="h-64 flex items-center justify-center text-muted-foreground bg-slate-50">
                                No preview
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground px-1">
                        <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> Processed in {usage.elapsed_time || '0.2'}s</span>
                        {usage.pages && <span>• {usage.pages} Page(s)</span>}
                    </div>
                </div>

                {/* RIGHT: Extracted Report */}
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-serif font-semibold text-foreground flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-accent" /> Intelligence Report
                        </h3>
                        <Button variant="ghost" size="sm" onClick={() => setShowJson(!showJson)} className="h-8 text-xs uppercase tracking-wider">
                            <Code className="w-3 h-3 mr-1.5" /> {showJson ? 'Table' : 'JSON'}
                        </Button>
                    </div>

                    <Card className="border-none shadow-lg shadow-slate-200/50 bg-white">
                        <CardContent className="p-0">
                            {showJson ? (
                                <div className="p-6 max-h-[500px] overflow-auto scrollbar-thin">
                                    <pre className="text-xs font-mono bg-slate-900 text-slate-50 p-4 rounded-md">
                                        {JSON.stringify(data, null, 2)}
                                    </pre>
                                </div>
                            ) : (
                                <div className="p-6">
                                    {/* Report Header */}
                                    <div className="flex justify-between items-end border-b border-border/60 pb-4 mb-4">
                                        <div>
                                            <h4 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Evaluation Metrics</h4>
                                            <p className="text-xs text-muted-foreground mt-1">Scale: 1 (Poor) to 4 (Excellent)</p>
                                        </div>
                                        <div className="text-right">
                                            <span className="text-2xl font-serif font-bold text-primary">
                                                {/* Simple Average? or just 'Agnel' logo? */}
                                                A+
                                            </span>
                                        </div>
                                    </div>

                                    {/* Rows */}
                                    <div className="space-y-1">
                                        {sortedKeys.map((key) => {
                                            const entry = displayFields[key]
                                            // Handle both {value:3} object and simple "3" value
                                            const val = typeof entry === 'object' ? entry.value : entry
                                            const conf = typeof entry === 'object' ? entry.confidence : 1.0

                                            // Use mapped label or fallback to key
                                            const label = LABELS[key] || key.replace(/_/g, ' ')

                                            return <ScoreRow key={key} label={label} value={val} confidence={conf} />
                                        })}

                                        {sortedKeys.length === 0 && (
                                            <div className="text-center py-8 text-muted-foreground italic">
                                                No metrics found.
                                            </div>
                                        )}
                                    </div>

                                    {/* Comments */}
                                    <CommentSection text={comments} />
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>
            </div>

            {/* Raw Data Accordion */}
            <div className="pt-8 border-t border-border/40">
                <button
                    onClick={() => setShowRaw(!showRaw)}
                    className="group flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                    <div className="p-1 rounded bg-slate-100 group-hover:bg-slate-200 transition-colors">
                        <ChevronRight className={cn("w-4 h-4 transition-transform duration-300", showRaw && "rotate-90")} />
                    </div>
                    <span>View Raw System Output</span>
                </button>

                {showRaw && (
                    <div className="mt-4 p-6 rounded-lg bg-slate-50 border border-border/50 animate-in slide-in-from-top-4">
                        <p className="text-xs font-mono text-muted-foreground mb-2">RAW_TEXT_ANNOTATION (Google Vision):</p>
                        <pre className="whitespace-pre-wrap text-xs text-slate-600 font-mono leading-relaxed max-h-[300px] overflow-auto">
                            {data.raw_text || rawText || "No raw text available."}
                        </pre>
                    </div>
                )}
            </div>
        </div>
    )
}
