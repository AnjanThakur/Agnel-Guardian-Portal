import React, { useCallback, useState } from 'react'
import { UploadCloud } from 'lucide-react'
import { cn } from '@/lib/utils'

export function FileUpload({ onFilesSelected }) {
    const [isDragging, setIsDragging] = useState(false)

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e) => {
        e.preventDefault()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback(
        (e) => {
            e.preventDefault()
            setIsDragging(false)
            const files = Array.from(e.dataTransfer.files)
            if (files.length > 0) {
                onFilesSelected(files)
            }
        },
        [onFilesSelected]
    )

    const handleFileInput = useCallback(
        (e) => {
            const files = Array.from(e.target.files)
            if (files.length > 0) {
                onFilesSelected(files)
            }
        },
        [onFilesSelected]
    )

    return (
        <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
                "relative flex flex-col items-center justify-center w-full min-h-[220px] border border-dashed rounded-xl transition-all duration-300 cursor-pointer bg-slate-50/50",
                isDragging
                    ? "border-primary bg-primary/5 scale-[0.99]"
                    : "border-slate-200 hover:border-primary/30 hover:bg-slate-50"
            )}
        >
            <input
                type="file"
                multiple
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                onChange={handleFileInput}
                accept="image/*,application/pdf"
            />
            <div className="flex flex-col items-center space-y-4 text-center p-8 pointer-events-none">
                <div className={cn(
                    "p-4 rounded-full bg-white shadow-sm ring-1 ring-slate-900/5 transition-transform duration-500",
                    isDragging ? "scale-110" : ""
                )}>
                    <UploadCloud className="w-8 h-8 text-primary/60" />
                </div>
                <div className="space-y-1">
                    <p className="text-lg font-serif font-medium text-foreground">
                        Drop files here
                    </p>
                    <p className="text-sm text-muted-foreground">
                        PDFs or Images
                    </p>
                </div>
            </div>
        </div>
    )
}
