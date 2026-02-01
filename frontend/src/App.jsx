import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FileUpload } from '@/components/FileUpload'
import { ResultsDisplay } from '@/components/ResultsDisplay'
import { SidebarList } from '@/components/SidebarList'
import { Loader2, ChevronDown, GraduationCap, ShieldCheck, RefreshCw, Trash2 } from 'lucide-react'

// Elegant Logo Component
const Logo = () => (
  <div className="flex items-center gap-2">
    <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary text-primary-foreground shadow-sm">
      <GraduationCap className="h-6 w-6" />
    </div>
  </div>
)

function App() {
  // --- STATE ---
  const [files, setFiles] = useState([]) // Array of File objects
  const [results, setResults] = useState({}) // Map: index -> { status: 'pending'|'loading'|'success'|'error', data: ..., error: ... }
  const [selectedIndex, setSelectedIndex] = useState(0) // Index of currently viewed file

  const [preview, setPreview] = useState(null) // Base64 of CURRENT viewed file
  const [mode, setMode] = useState('pta_free')
  const [template, setTemplate] = useState('')
  const [debug, setDebug] = useState(false)

  const [batchProcessing, setBatchProcessing] = useState(false)

  // --- HANDLER: File Selection ---
  const handleFilesSelected = (newFiles) => {
    // Append or Replace? Let's Replace for simplicity to start (clear old).
    // Or Append? "Inbox" implies appending. Let's start with Replace/Set.
    // If user drops new files, we reset.
    setFiles(newFiles)

    // Initialize results state
    const initialResults = {}
    newFiles.forEach((_, idx) => {
      initialResults[idx] = { status: 'pending' }
    })
    setResults(initialResults)

    setSelectedIndex(0)
    // We will load preview for index 0 via effect or manual call
  }

  // --- EFFECT: Load Preview when Selected Index changes ---
  useEffect(() => {
    if (files.length > 0 && files[selectedIndex]) {
      const file = files[selectedIndex]
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
    } else {
      setPreview(null)
    }
  }, [selectedIndex, files])


  // --- LOGIC: Helper to process ONE file ---
  const processSingleFile = async (file, index) => {
    try {
      // 1. Read File to Base64 (Promise wrapper)
      const toBase64 = (f) => new Promise((resolve, reject) => {
        const r = new FileReader()
        r.readAsDataURL(f)
        r.onload = () => resolve(r.result)
        r.onerror = error => reject(error)
      })
      const base64 = await toBase64(file)

      // 2. Prepare Payload
      const url =
        mode === "pta"
          ? "/ocr/pta"
          : mode === "pta_free"
            ? "/ocr/pta_free"
            : "/ocr/auto"

      const payload = {
        imageBase64: base64, // The API expects raw field name 'imageBase64'
        template: template || null,
        debug: debug,
      }

      // 3. Update Status -> Loading
      setResults(prev => ({
        ...prev,
        [index]: { ...prev[index], status: 'loading' }
      }))

      // 4. API Call
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      // 5. Update Status -> Success
      setResults(prev => ({
        ...prev,
        [index]: { status: 'success', data: data }
      }))

    } catch (err) {
      console.error(`Error processing file ${index}:`, err)

      // 6. Update Status -> Error
      setResults(prev => ({
        ...prev,
        [index]: { status: 'error', error: err.message || "Failed" }
      }))
    }
  }

  // --- LOGIC: Batch Loop ---
  const runBatchOCR = async () => {
    setBatchProcessing(true)

    // Sequential Loop
    for (let i = 0; i < files.length; i++) {
      // Skip if already successfully processed? 
      // Optional: comment out next line to force re-process all
      if (results[i]?.status === 'success') continue;

      // Scroll Sidebar to active? (Optional UX)
      // setSelectedIndex(i) // Auto-switch view? Maybe annoying. Let's NOT auto-switch view, just update status.

      await processSingleFile(files[i], i)
    }

    setBatchProcessing(false)
  }

  const clearAll = () => {
    setFiles([])
    setResults({})
    setSelectedIndex(0)
    setPreview(null)
  }

  // Computed for UI
  const currentResult = results[selectedIndex] || {}
  const hasFiles = files.length > 0

  return (
    <div className="min-h-screen bg-background font-sans text-foreground selection:bg-primary/10">

      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
        <div className="container flex h-20 items-center justify-between px-4 md:px-8 max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <Logo />
            <div className="flex flex-col">
              <span className="text-xl font-serif font-bold tracking-tight text-primary">Agnel Guardian</span>
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8 md:py-12 space-y-8">

        {!hasFiles ? (
          // --- VIEW 1: EMPTY STATE (HERO UPLOAD) ---
          <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="text-center space-y-4">
              <h1 className="text-4xl font-serif font-medium text-primary tracking-tight">
                Start Extraction
              </h1>
              <p className="text-muted-foreground text-lg font-light">
                Upload batches of PTA forms, PDF transcripts, or receipts.
              </p>
            </div>

            <Card className="border-none shadow-2xl shadow-slate-200/50">
              <FileUpload onFilesSelected={handleFilesSelected} />
            </Card>

            {/* Quick Config (Optional for empty state) */}
            <div className="flex justify-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-primary" /> Secure Processing
              </div>
              <div className="flex items-center gap-2">
                <RefreshCw className="w-4 h-4 text-primary" /> Auto-Scaling Logic
              </div>
            </div>
          </div>
        ) : (
          // --- VIEW 2: WORKSPACE (INBOX GRID) ---
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start animate-in fade-in duration-500">

            {/* --- LEFT COLUMN: CONFIG + INBOX --- */}
            <div className="lg:col-span-4 space-y-6 flex flex-col h-[calc(100vh-140px)] sticky top-24">

              {/* Action Bar */}
              <div className="flex items-center gap-2">
                <Button
                  size="lg"
                  onClick={runBatchOCR}
                  disabled={batchProcessing}
                  className="flex-1 font-serif tracking-wide shadow-lg shadow-primary/20"
                >
                  {batchProcessing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {batchProcessing ? 'Processing Batch...' : 'Run Extraction'}
                </Button>
                <Button variant="outline" size="icon" onClick={clearAll} disabled={batchProcessing} title="Clear All">
                  <Trash2 className="w-4 h-4 text-muted-foreground" />
                </Button>
              </div>

              {/* Config Card (Compact) */}
              <Card className="border-none shadow-md bg-white/60 backdrop-blur-sm shrink-0">
                <CardHeader className="py-3 px-4">
                  <CardTitle className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Settings</CardTitle>
                </CardHeader>
                <CardContent className="pb-4 px-4 space-y-3">
                  <div className="space-y-1">
                    <div className="relative">
                      <select
                        className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring appearance-none"
                        value={mode}
                        onChange={(e) => setMode(e.target.value)}
                      >
                        <option value="auto">Intelligent (Auto)</option>
                        <option value="pta">Template (YAML)</option>
                        <option value="pta_free">PTA (Label-based)</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-2.5 h-3 w-3 text-muted-foreground pointer-events-none opacity-50" />
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="debug"
                      className="w-3.5 h-3.5 rounded-sm border-primary text-primary focus:ring-primary"
                      checked={debug}
                      onChange={(e) => setDebug(e.target.checked)}
                    />
                    <Label htmlFor="debug" className="cursor-pointer font-normal text-xs text-muted-foreground">Debug Mode</Label>
                  </div>
                </CardContent>
              </Card>

              {/* Sidebar Inbox List (Scrollable) */}
              <div className="flex-1 min-h-0">
                <SidebarList
                  files={files}
                  activeIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                  results={results}
                />
              </div>

            </div>


            {/* --- RIGHT COLUMN: RESULT DETAIL --- */}
            <div className="lg:col-span-8 space-y-6">
              {/* Detail View for Selected File */}
              <div className="min-h-[600px]">
                <ResultsDisplay
                  data={currentResult.data || {}}
                  previewUrl={preview}
                // If we have an error, we can pass it or handle it in ResultsDisplay? 
                // ResultsDisplay expects 'data'. Let's handle 'error' roughly here for now.
                />

                {currentResult.status === 'error' && (
                  <div className="mt-4 p-4 border border-destructive/20 bg-destructive/5 rounded-lg text-destructive text-sm font-medium">
                    Analysis Failed: {currentResult.error}
                  </div>
                )}

                {currentResult.status === 'pending' && !currentResult.error && (
                  <div className="h-64 flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed border-slate-100 rounded-xl bg-slate-50/50 mt-4">
                    <p className="font-serif italic">Ready to process.</p>
                    <p className="text-xs mt-1">Click "Run Extraction" to start batch.</p>
                  </div>
                )}

                {currentResult.status === 'loading' && (
                  <div className="h-64 flex flex-col items-center justify-center text-primary mt-4">
                    <Loader2 className="w-8 h-8 animate-spin mb-4 text-accent" />
                    <p className="font-serif">Analyzing document...</p>
                  </div>
                )}
              </div>
            </div>

          </div>
        )}

      </main>
    </div>
  )
}

export default App
