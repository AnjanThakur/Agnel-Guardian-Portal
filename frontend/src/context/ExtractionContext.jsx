import React, { createContext, useContext, useState } from 'react';

const ExtractionContext = createContext();

export const ExtractionProvider = ({ children }) => {
    const [files, setFiles] = useState([]);
    const [results, setResults] = useState({});
    const [aiSummary, setAiSummary] = useState(null);
    const [selectedIndex, setSelectedIndex] = useState(0);

    const clearAll = () => {
        setFiles([]);
        setResults({});
        setAiSummary(null);
        setSelectedIndex(0);
    };

    return (
        <ExtractionContext.Provider value={{
            files, setFiles,
            results, setResults,
            aiSummary, setAiSummary,
            selectedIndex, setSelectedIndex,
            clearAll
        }}>
            {children}
        </ExtractionContext.Provider>
    );
};

export const useExtraction = () => {
    const context = useContext(ExtractionContext);
    if (!context) {
        throw new Error('useExtraction must be used within an ExtractionProvider');
    }
    return context;
};
