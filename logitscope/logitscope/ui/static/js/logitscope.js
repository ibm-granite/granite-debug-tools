/**
 * LogitScope UI - Main JavaScript Module
 *
 * Provides token-level analysis and visualization for language model outputs.
 * Integrates with backend API via WebSocket for real-time analysis.
 *
 * @module entroscope
 */

import { CONFIG } from '/static/js/config.js';

// Debug Flag - controlled by configuration
const DEBUG = CONFIG.DEBUG;

/**
 * TokenAnalyzer - Manages token analysis and backend communication
 *
 * Responsibilities:
 * - WebSocket connection management for real-time analysis
 * - Token parsing and position tracking
 * - Backend API communication
 * - State management (undo/redo)
 * - Metric validation and sanitization
 *
 * @class
 */
class TokenAnalyzer {
    /**
     * Initialize the token analyzer
     * Sets up WebSocket connection and fetches model info
     */
    constructor() {
        /** @type {Array} Analyzed tokens with metrics */
        this.tokens = [];

        /** @type {number} Currently selected token index */
        this.selectedTokenIndex = CONFIG.STATE.DEFAULT_SELECTED_INDEX;

        /** @type {Array} Undo history stack */
        this.undoStack = [];

        /** @type {Array} Redo history stack */
        this.redoStack = [];

        /** @type {number} Maximum undo steps to retain */
        this.maxUndoSteps = CONFIG.ANALYSIS.MAX_UNDO_STEPS;

        /** @type {boolean} Whether analysis is currently in progress */
        this.isAnalyzing = false;

        // Performance optimization
        /** @type {string} Last analyzed text for caching */
        this.lastAnalyzedText = '';

        /** @type {Array} Cached token results */
        this.cachedTokens = [];

        // WebSocket connection for real-time analysis
        /** @type {WebSocket|null} WebSocket connection */
        this.websocket = null;

        /** @type {boolean} WebSocket connection status */
        this.websocketConnected = false;

        /** @type {string|null} Pending analysis request text */
        this.pendingAnalysisRequest = null;

        /** @type {number|null} Analysis debounce timer */
        this.analysisDebounceTimer = null;

        /** @type {number} Debounce delay in milliseconds */
        this.analysisDebounceDelay = CONFIG.ANALYSIS.DEBOUNCE_DELAY;

        /** @type {number} Last request ID for handling out-of-order responses */
        this.lastRequestId = 0;

        this.connectWebSocket();

        // Fetch model info immediately on initialization
        this.fetchModelInfo();
    }

    // Tokenize text preserving whitespace and punctuation with precise positioning
    tokenize(text) {
        if (!text) return [];

        const tokens = [];
        let currentToken = '';
        let tokenStart = 0;

        for (let i = 0; i <= text.length; i++) {
            const char = i < text.length ? text[i] : null;
            const isWhitespace = char && /\s/.test(char);
            const isPunctuation = char && /[.!?,:;(){}[\]"']/.test(char);
            const isWordChar = char && !isWhitespace && !isPunctuation;

            // If we're at a boundary (whitespace, punctuation, or end of text)
            if (!isWordChar) {
                // Save the current word token if we have one
                if (currentToken) {
                    tokens.push({
                        text: currentToken,
                        start: tokenStart,
                        end: i,
                        index: tokens.length,
                        isWord: true
                    });
                    currentToken = '';
                }

                // Handle the current character if it exists
                if (char !== null) {
                    if (isWhitespace) {
                        // Handle different types of whitespace
                        let whitespaceText = char;
                        let whitespaceEnd = i + 1;

                        // Handle Windows line endings \r\n
                        if (char === '\r' && i + 1 < text.length && text[i + 1] === '\n') {
                            whitespaceText = '\r\n';
                            whitespaceEnd = i + 2;
                            i++; // Skip the \n in the next iteration
                        }

                        tokens.push({
                            text: whitespaceText,
                            start: i - (whitespaceText.length - 1),
                            end: whitespaceEnd,
                            index: tokens.length,
                            isWhitespace: true,
                            isNewline: /[\r\n]/.test(whitespaceText)
                        });

                        tokenStart = whitespaceEnd;
                    } else if (isPunctuation) {
                        tokens.push({
                            text: char,
                            start: i,
                            end: i + 1,
                            index: tokens.length,
                            isPunctuation: true
                        });
                        tokenStart = i + 1;
                    }
                }
            } else {
                // We're building a word token
                if (currentToken === '') {
                    tokenStart = i;
                }
                currentToken += char;
            }
        }

        return tokens;
    }

    // WebSocket connection management
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                this.websocketConnected = true;
                if (DEBUG) {
                    console.log('WebSocket connected for real-time analysis');
                }
                this.updateConnectionStatus('connected');
                
                // Retry fetching model info if not already loaded
                const modelElement = document.getElementById('currentModel');
                if (modelElement && (modelElement.textContent === 'Loading...' || 
                                   modelElement.textContent === 'Connecting...' || 
                                   modelElement.textContent === 'GPT-2 Base')) {
                    this.fetchModelInfo();
                }
                
                // Send ping to verify connection
                this.sendWebSocketMessage({
                    type: 'ping'
                });
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };
            
            this.websocket.onclose = () => {
                this.websocketConnected = false;
                if (DEBUG) {
                    console.log('WebSocket disconnected');
                }
                this.updateConnectionStatus('disconnected');

                // Attempt to reconnect
                setTimeout(() => {
                    this.connectWebSocket();
                }, CONFIG.WEBSOCKET.RECONNECT_DELAY);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.websocketConnected = false;
                this.updateConnectionStatus('error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.websocketConnected = false;
            this.updateConnectionStatus('error');
        }
    }

    updateConnectionStatus(status) {
        // Update the analysis status indicator
        const statusElement = document.getElementById('analysis-status');
        if (statusElement) {
            switch (status) {
                case 'connected':
                    statusElement.textContent = 'Connected';
                    statusElement.className = 'stat-value status-connected';
                    break;
                case 'disconnected':
                    statusElement.textContent = 'Disconnected';
                    statusElement.className = 'stat-value status-disconnected';
                    break;
                case 'error':
                    statusElement.textContent = 'Error';
                    statusElement.className = 'stat-value status-error';
                    break;
                case 'analyzing':
                    statusElement.textContent = 'Analyzing...';
                    statusElement.className = 'stat-value status-analyzing';
                    break;
                default:
                    statusElement.textContent = 'Ready';
                    statusElement.className = 'stat-value';
            }
        }
    }

    updateModelName(modelName) {
        // Update the model name display in the left panel
        const modelElement = document.getElementById('currentModel');
        if (modelElement && modelName) {
            // Clean up the model name for display (remove path prefixes, etc.)
            let displayName = modelName;
            
            // Extract just the model name from paths like "microsoft/DialoGPT-medium"
            if (displayName.includes('/')) {
                const parts = displayName.split('/');
                displayName = parts[parts.length - 1];
            }
            
            // Capitalize and format common model names using configuration
            const lowerName = displayName.toLowerCase();
            for (const [pattern, replacement] of Object.entries(CONFIG.MODEL_NAMES)) {
                if (lowerName === pattern || lowerName.startsWith(pattern)) {
                    displayName = displayName.replace(new RegExp(`^${pattern}`, 'i'), replacement);
                    break;
                }
            }
            
            modelElement.textContent = displayName;
            if (DEBUG) {
                console.log(`Model name updated to: ${displayName}`);
            }
        }
    }

    async fetchModelInfo() {
        try {
            const response = await fetch('/model/info');
            if (response.ok) {
                const modelInfo = await response.json();
                if (modelInfo.model_name) {
                    this.updateModelName(modelInfo.model_name);
                }
                if (DEBUG) {
                    console.log('Model info loaded:', modelInfo);
                }
            } else if (response.status === 503) {
                if (DEBUG) {
                    console.log('Model not yet loaded, will retry when WebSocket connects');
                }
                // Set a placeholder that indicates loading
                const modelElement = document.getElementById('currentModel');
                if (modelElement) {
                    modelElement.textContent = 'Loading...';
                }
            }
        } catch (error) {
            console.log('Failed to fetch model info:', error);
            // Try again when WebSocket connects
            const modelElement = document.getElementById('currentModel');
            if (modelElement) {
                modelElement.textContent = 'Connecting...';
            }
        }
    }

    sendWebSocketMessage(message) {
        if (this.websocket && this.websocketConnected) {
            try {
                this.websocket.send(JSON.stringify(message));
            } catch (error) {
                console.error('Failed to send WebSocket message:', error);
            }
        }
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'pong':
                // Connection verified
                break;
            
            case 'analysis_result':
                // Check if this response is for the latest request
                if (message.requestId && message.requestId === this.lastRequestId) {
                    this.handleAnalysisResult(message.data);
                } else if (!message.requestId) {
                    // Handle legacy responses without requestId
                    this.handleAnalysisResult(message.data);
                }
                break;
            
            case 'error':
                console.error('Backend analysis error:', message.message);
                this.handleBackendUnavailable();
                break;
            
            default:
                if (DEBUG) {
                    console.log('Unknown WebSocket message type:', message.type);
                }
        }
    }

    handleAnalysisResult(analysisData) {
        try {
            // Convert backend response to our token format
            const backendTokens = analysisData.tokens || [];

            if (DEBUG && backendTokens.length > 0) {
                console.log('Backend analysis result sample:', {
                    totalTokens: backendTokens.length,
                    firstToken: backendTokens[0],
                    metricsAvailable: backendTokens[0]?.metrics ? Object.keys(backendTokens[0].metrics) : []
                });
            }

            // Update model name display if provided
            if (analysisData.model_name) {
                this.updateModelName(analysisData.model_name);
            }

            // Match backend tokens with original text positions
            const originalText = this.lastAnalyzedText;
            this.tokens = this.mapBackendTokensToTextPositions(backendTokens, originalText);

            this.isAnalyzing = false;
            this.pendingAnalysisRequest = null;
            this.updateConnectionStatus('connected');

            // Cache the results
            this.cachedTokens = [...this.tokens];

            // Validate token positions for debugging
            this.validateTokenPositions();

            // Trigger immediate UI update for real-time feel
            if (window.entroscope) {
                window.entroscope.updateVisualization();
                window.entroscope.updateLiveStatistics();
            }

        } catch (error) {
            console.error('Error processing analysis result:', error);
            this.handleBackendUnavailable();
        }
    }

    mapBackendTokensToTextPositions(backendTokens, originalText) {
        // This function maps backend tokens to their actual positions in the original text
        const mappedTokens = [];
        let textPosition = 0;
        
        for (let i = 0; i < backendTokens.length; i++) {
            const backendToken = backendTokens[i];
            const metrics = backendToken.metrics || {};
            const rawTokenText = backendToken.token;
            const decodedTokenText = this.decodeTokenizerString(rawTokenText);
            
            // The key insight: the original text contains what the user typed
            // The backend tokenizer returns raw tokens, but we need to map to decoded tokens
            // because that's what should appear in the original text
            
            // Find the decoded token in the original text starting from current position
            let foundPosition = originalText.indexOf(decodedTokenText, textPosition);
            
            // If not found at current position, try to find it nearby (handles tokenization differences)
            if (foundPosition === -1) {
                // Search within a small range around current position
                const searchStart = Math.max(0, textPosition - 5);
                const searchEnd = Math.min(originalText.length, textPosition + 20);
                const searchSubstring = originalText.substring(searchStart, searchEnd);
                const relativePos = searchSubstring.indexOf(decodedTokenText);
                
                if (relativePos !== -1) {
                    foundPosition = searchStart + relativePos;
                }
            }
            
            // If still not found, use sequential position (fallback)
            if (foundPosition === -1) {
                foundPosition = textPosition;
            }
            
            const startPos = foundPosition;
            const endPos = foundPosition + decodedTokenText.length;
            textPosition = endPos;
            
            mappedTokens.push({
                text: decodedTokenText, // Use decoded text for display and positioning
                rawText: rawTokenText, // Store original tokenizer string for reference
                start: startPos,
                end: endPos,
                index: backendToken.index,
                token_id: backendToken.token_id,
                
                // All metrics directly from backend (validated and cleaned)
                entropy: this.validateMetricValue(metrics.entropy, 'entropy', 0),
                varentropy: this.validateMetricValue(metrics.varentropy, 'varentropy', 0),
                skewentropy: this.validateMetricValue(metrics.skewentropy, 'skewentropy', 0),
                perplexity: this.validateMetricValue(metrics.perplexity, 'perplexity', 1),
                surprisal: this.validateMetricValue(metrics.surprisal, 'surprisal', 0),
                
                // Probability metrics (derived from top-k candidates or computed)
                probability: this.validateMetricValue(this.extractTokenProbability(backendToken), 'probability', 0),
                logProbability: this.validateMetricValue(this.extractTokenLogProbability(backendToken), 'logProbability', -10),
                
                // Top-k candidates from backend
                top_k_candidates: backendToken.top_k_candidates || [],
                
                // Token type flags (based on decoded text)
                isWhitespace: /^\s+$/.test(decodedTokenText),
                isPunctuation: /^[.!?,:;(){}[\]"']+$/.test(decodedTokenText),
                isWord: /^[a-zA-Z0-9_]+$/.test(decodedTokenText)
            });
        }
        
        return mappedTokens;
    }

    /**
     * Decode HuggingFace tokenizer string artifacts
     * Handles various tokenizer-specific string encodings and special characters
     * @param {string} tokenText - Raw token text from tokenizer
     * @returns {string} Decoded token text
     */
    decodeTokenizerString(tokenText) {
        if (!tokenText) return tokenText;

        // Handle GPT-style Ġ prefix (represents space)
        if (tokenText.startsWith(CONFIG.TOKENIZER.GPT_SPACE_PREFIX)) {
            return ' ' + tokenText.substring(1);
        }

        // Handle T5/BERT-style ▁ prefix (represents space)
        if (tokenText.startsWith(CONFIG.TOKENIZER.T5_SPACE_PREFIX)) {
            return ' ' + tokenText.substring(1);
        }

        // Handle GPT-style special characters
        let decoded = tokenText;

        // Handle Ċ (C with dot above) - represents newline in GPT tokenizers
        decoded = decoded.replace(new RegExp(CONFIG.TOKENIZER.GPT_NEWLINE_CHAR, 'g'), '\n');
        
        // Handle Ġ in the middle of tokens (space)
        decoded = decoded.replace(/Ġ/g, ' ');
        
        // Handle other common tokenizer artifacts
        // Handle newline representations
        decoded = decoded.replace(/\<\|newline\|\>/g, '\n');
        decoded = decoded.replace(/\<n\>/g, '\n');
        decoded = decoded.replace(/\\n/g, '\n');
        
        // Handle tab representations  
        decoded = decoded.replace(/\<\|tab\|\>/g, '\t');
        decoded = decoded.replace(/\<t\>/g, '\t');
        decoded = decoded.replace(/\\t/g, '\t');
        
        // Handle UTF-8 encoding artifacts from BPE tokenizers
        for (const [artifact, replacement] of Object.entries(CONFIG.TOKENIZER.BPE_ARTIFACTS)) {
            decoded = decoded.replace(new RegExp(artifact, 'g'), replacement);
        }
        
        // Handle other escape sequences
        decoded = decoded.replace(/\\r/g, '\r');
        decoded = decoded.replace(/\\"/g, '"');
        decoded = decoded.replace(/\\'/g, "'");
        decoded = decoded.replace(/\\\\/g, '\\');
        
        return decoded;
    }

    // Validate and clean metric values to handle NaN, Infinity, etc.
    validateMetricValue(value, metricName, fallbackValue = 0) {
        // Convert to number if it's a string
        const numValue = typeof value === 'string' ? parseFloat(value) : value;

        // Check for invalid values
        if (numValue === null || numValue === undefined) {
            if (DEBUG) {
                console.warn(`Metric ${metricName} is null/undefined, using fallback: ${fallbackValue}`);
            }
            return fallbackValue;
        }

        if (isNaN(numValue)) {
            if (DEBUG) {
                console.warn(`Metric ${metricName} is NaN, using fallback: ${fallbackValue}`);
            }
            return fallbackValue;
        }
        
        if (!isFinite(numValue)) {
            if (numValue === Infinity) {
                if (DEBUG) {
                    console.warn(`Metric ${metricName} is Infinity, clamping to safe maximum`);
                }
                // Use reasonable maximums from configuration
                switch (metricName) {
                    case 'entropy': return CONFIG.METRICS.ENTROPY_MAX;
                    case 'varentropy': return CONFIG.METRICS.VARENTROPY_MAX;
                    case 'skewentropy': return CONFIG.METRICS.SKEWENTROPY_MAX;
                    case 'perplexity': return CONFIG.METRICS.PERPLEXITY_MAX;
                    case 'surprisal': return CONFIG.METRICS.SURPRISAL_MAX;
                    default: return CONFIG.METRICS.DEFAULT_MAX;
                }
            } else if (numValue === -Infinity) {
                if (DEBUG) {
                    console.warn(`Metric ${metricName} is -Infinity, using fallback: ${fallbackValue}`);
                }
                return fallbackValue;
            }
        }

        // Additional sanity checks for specific metrics using configuration
        switch (metricName) {
            case 'perplexity':
                return Math.max(numValue, CONFIG.METRICS.PERPLEXITY_MIN);
            case 'probability':
                return Math.max(CONFIG.METRICS.PROBABILITY_MIN, Math.min(CONFIG.METRICS.PROBABILITY_MAX, numValue));
            case 'logProbability':
                return Math.max(CONFIG.METRICS.LOG_PROB_MIN, Math.min(CONFIG.METRICS.LOG_PROB_MAX, numValue));
            default:
                return numValue;
        }
    }

    // Extract token probability from top-k candidates (if available)
    extractTokenProbability(backendToken) {
        // If the token has top-k candidates, the first one should be the actual token
        if (backendToken.top_k_candidates && backendToken.top_k_candidates.length > 0) {
            // Find the candidate that matches this token
            const matchingCandidate = backendToken.top_k_candidates.find(
                candidate => candidate.token === backendToken.token
            );
            if (matchingCandidate) {
                return matchingCandidate.probability;
            }
            // If no exact match, use the first candidate as approximation
            return backendToken.top_k_candidates[0].probability;
        }
        return 0;
    }

    // Extract token log probability
    extractTokenLogProbability(backendToken) {
        const probability = this.extractTokenProbability(backendToken);
        return probability > 0 ? Math.log(probability) : -10;
    }

    // Build position map for textarea click handling
    // This should map based on the original textarea positions, not raw tokenizer text
    buildTextareaPositionMap() {
        const positionMap = [];
        
        for (let i = 0; i < this.tokens.length; i++) {
            const token = this.tokens[i];
            
            positionMap.push({
                tokenIndex: i,
                start: token.start,  // Use the original positions calculated in mapBackendTokensToTextPositions
                end: token.end,      // These should correspond to the textarea content
                rawText: token.rawText,
                decodedText: token.text
            });
        }
        
        return positionMap;
    }

    validateTokenPositions() {
        if (!DEBUG) return;
        
        // Reconstruct text from tokens to verify positions
        let reconstructedText = '';
        let expectedPosition = 0;
        
        for (let i = 0; i < this.tokens.length; i++) {
            const token = this.tokens[i];
            
            // Check if token starts where expected
            if (token.start !== expectedPosition) {
                if (DEBUG) {
                    console.warn(`Token ${i} position mismatch: expected start ${expectedPosition}, got ${token.start}. Token: "${token.text}"`);
                }
            }

            // Check if the token text matches the original text at this position
            const actualText = this.lastAnalyzedText.substring(token.start, token.end);
            if (actualText !== token.text) {
                if (DEBUG) {
                    console.warn(`Token ${i} text mismatch at position ${token.start}-${token.end}: expected "${token.text}", got "${actualText}"`);
                }
            }
            
            reconstructedText += token.text;
            expectedPosition = token.end;
        }
        
        // Check if reconstructed text matches original
        if (reconstructedText !== this.lastAnalyzedText) {
            if (DEBUG) {
                console.warn('Reconstructed text does not match original text');
                console.log('Original:', JSON.stringify(this.lastAnalyzedText));
                console.log('Reconstructed:', JSON.stringify(reconstructedText));
            }
        } else {
            if (DEBUG) {
                console.log('Token positions validated successfully');
            }
        }
    }

    handleBackendUnavailable() {
        if (DEBUG) {
            console.warn('Backend unavailable - clearing token data');
        }
        this.tokens = [];
        this.cachedTokens = [];
        this.isAnalyzing = false;
        this.updateConnectionStatus('disconnected');
    }

    // Analyze text and generate tokens with metrics
    analyzeText(text, forceRealTime = false) {
        // For real-time mode, skip cache for small changes to provide immediate feedback
        if (!forceRealTime && text === this.lastAnalyzedText && this.cachedTokens.length > 0) {
            this.tokens = this.cachedTokens;
            return this.tokens;
        }

        this.lastAnalyzedText = text;

        // If text is empty, clear tokens
        if (!text || text.trim() === '') {
            this.tokens = [];
            this.cachedTokens = [];
            this.isAnalyzing = false;
            return this.tokens;
        }

        // Only proceed if backend is connected
        if (this.websocketConnected) {
            this.requestBackendAnalysis(text);
        } else {
            if (DEBUG) {
                console.warn('Backend not available - no analysis performed');
            }
            this.handleBackendUnavailable();
        }

        return this.tokens;
    }

    /**
     * Request backend analysis with adaptive debouncing
     * Delays are adjusted based on text length and input patterns
     * @param {string} text - Text to analyze
     */
    requestBackendAnalysis(text) {
        // Adaptive debouncing based on text characteristics
        const isShortText = text.length < CONFIG.ANALYSIS.SHORT_TEXT_LENGTH;
        const endsWithSpace = text.endsWith(' ') || text.endsWith('\n') || text.endsWith('\t');
        const endsWithPunctuation = /[.!?,:;]$/.test(text);

        let delay = this.analysisDebounceDelay;
        if (isShortText || endsWithSpace || endsWithPunctuation) {
            delay = 0; // Immediate for short text or word boundaries
        } else if (text.length < CONFIG.ANALYSIS.FAST_TEXT_LENGTH) {
            delay = CONFIG.ANALYSIS.DEBOUNCE_DELAY_FAST;
        }
        
        // Clear any existing debounce timer
        if (this.analysisDebounceTimer) {
            clearTimeout(this.analysisDebounceTimer);
        }

        // Debounce the analysis requests (or send immediately for certain conditions)
        this.analysisDebounceTimer = setTimeout(() => {
            this.isAnalyzing = true;
            this.updateConnectionStatus('analyzing');
            
            // Generate unique request ID to handle out-of-order responses
            this.lastRequestId++;
            const requestId = this.lastRequestId;
            
            // Cancel any pending request
            this.pendingAnalysisRequest = text;
            
            // Send analysis request via WebSocket
            this.sendWebSocketMessage({
                type: 'analyze',
                requestId: requestId,
                data: {
                    text: text,
                    metrics: ['entropy', 'varentropy', 'skewentropy', 'perplexity', 'surprisal'],
                    top_k: CONFIG.ANALYSIS.DEFAULT_TOP_K,
                    generate: false
                }
            });
        }, delay);
    }

    // Save state for undo functionality
    saveState(textarea) {
        const state = {
            text: textarea.value,
            cursorPosition: textarea.selectionStart,
            tokens: JSON.parse(JSON.stringify(this.tokens)),
            selectedTokenIndex: this.selectedTokenIndex
        };

        this.undoStack.push(state);
        if (this.undoStack.length > this.maxUndoSteps) {
            this.undoStack.shift();
        }
        this.redoStack = []; // Clear redo stack when new action is performed
    }

    // Undo last action
    undo(textarea) {
        if (this.undoStack.length === 0) return false;

        const currentState = {
            text: textarea.value,
            cursorPosition: textarea.selectionStart,
            tokens: JSON.parse(JSON.stringify(this.tokens)),
            selectedTokenIndex: this.selectedTokenIndex
        };

        this.redoStack.push(currentState);

        const previousState = this.undoStack.pop();
        textarea.value = previousState.text;
        textarea.setSelectionRange(previousState.cursorPosition, previousState.cursorPosition);
        this.tokens = previousState.tokens;
        this.selectedTokenIndex = previousState.selectedTokenIndex;

        return true;
    }

    // Redo last undone action
    redo(textarea) {
        if (this.redoStack.length === 0) return false;

        const currentState = {
            text: textarea.value,
            cursorPosition: textarea.selectionStart,
            tokens: JSON.parse(JSON.stringify(this.tokens)),
            selectedTokenIndex: this.selectedTokenIndex
        };

        this.undoStack.push(currentState);

        const nextState = this.redoStack.pop();
        textarea.value = nextState.text;
        textarea.setSelectionRange(nextState.cursorPosition, nextState.cursorPosition);
        this.tokens = nextState.tokens;
        this.selectedTokenIndex = nextState.selectedTokenIndex;

        return true;
    }

    // Select token by index
    selectToken(index) {
        this.selectedTokenIndex = index;
    }

    /**
     * Get top-k candidate tokens for a position
     * Returns candidates from backend data only (no mock data)
     * @param {number} tokenIndex - Index of token in tokens array
     * @param {number} k - Number of candidates to return (default from config)
     * @returns {Array} Array of candidate objects with token, probability, and rank
     */
    getTopKCandidates(tokenIndex, k = CONFIG.ANALYSIS.DEFAULT_TOP_K) {
        const selectedToken = this.tokens[tokenIndex];
        if (!selectedToken) return [];

        // Only return candidates if we have real backend data
        if (selectedToken.top_k_candidates && selectedToken.top_k_candidates.length > 0) {
            return selectedToken.top_k_candidates.slice(0, k).map((candidate, index) => ({
                token: candidate.token,
                probability: candidate.probability,
                rank: index + 1
            }));
        }

        // Return empty array if no backend data available
        return [];
    }

    // Export data as JSON
    exportData() {
        return {
            tokens: this.tokens,
            statistics: this.getStatistics(),
            timestamp: new Date().toISOString()
        };
    }

    // Get overall statistics
    getStatistics() {
        // Return default statistics when there are no tokens
        if (this.tokens.length === 0) {
            return {
                totalTokens: 0,
                wordTokens: 0,
                characters: 0,
                avgEntropy: 0,
                avgVariance: 0,
                avgSkewentropy: 0,
                avgPerplexity: 1,
                avgProbability: 0,
                avgLogProb: -10,
                avgAttention: 0
            };
        }

        const validTokens = this.tokens.filter(t => !t.isWhitespace);

        // Safely calculate averages with validation
        const safeAverage = (values, fallback = 0) => {
            if (values.length === 0) return fallback;
            const validValues = values.filter(v => isFinite(v) && !isNaN(v));
            if (validValues.length === 0) return fallback;
            return validValues.reduce((sum, v) => sum + v, 0) / validValues.length;
        };

        return {
            totalTokens: this.tokens.length,
            wordTokens: validTokens.length,
            characters: this.tokens.reduce((sum, t) => sum + t.text.length, 0),
            avgEntropy: safeAverage(validTokens.map(t => t.entropy), 0),
            avgVariance: safeAverage(validTokens.map(t => t.varentropy), 0),
            avgSkewentropy: safeAverage(validTokens.map(t => t.skewentropy), 0),
            avgPerplexity: safeAverage(validTokens.map(t => t.perplexity), 1),
            avgProbability: safeAverage(validTokens.map(t => t.probability), 0),
            avgLogProb: safeAverage(validTokens.map(t => t.logProbability), -10),
            avgAttention: safeAverage(validTokens.map(t => t.attention || 0), 0)
        };
    }
}


/**
 * Main LogitScope UI Controller
 * Manages the visualization interface and coordinates with TokenAnalyzer
 */
class LogitScope {
    constructor() {
        this.tokenAnalyzer = new TokenAnalyzer();
        this.activeTab = CONFIG.STATE.DEFAULT_TAB;

        // Performance optimization properties
        this.renderDebounceTimer = null;
        this.lastProcessedText = '';
        this.isRendering = false;
        this.cachedMetrics = null;
        this.metricsChangeObserver = null;

        this.initializeElements();
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        this.initializePlots();

        // Set initial tab state
        this.switchTab(CONFIG.STATE.DEFAULT_TAB);

        // Initialize legend visibility
        this.updateLegendVisibility();

        // Save initial state
        setTimeout(() => {
            this.tokenAnalyzer.saveState(this.textarea);
        }, 100);
    }

    initializeElements() {
        this.textarea = document.getElementById('canvasTextArea');
        this.tokenDisplay = document.getElementById('tokenDisplay');
        this.tokenExplorer = document.getElementById('tokenExplorer');
        this.explorerLeftPanel = document.querySelector('.explorer-left-panel');
        this.explorerRightPanel = document.querySelector('.explorer-right-panel');
        this.tokenCardsContainer = document.getElementById('tokenCardsContainer');
        this.topkCandidates = document.getElementById('topkCandidates');
        this.distributionsContent = document.getElementById('distributionsContent');

        // Top-k candidates elements
        this.candidatesPanel = document.getElementById('candidates-panel');
        this.candidatesList = document.getElementById('candidates-list');
        this.candidatePosition = document.getElementById('candidate-position');
        this.kValueInput = document.getElementById('k-value-input');

        // Token explorer elements
        this.tokenList = document.getElementById('token-list');
        this.scrollLeftBtn = document.getElementById('scroll-left-btn');
        this.scrollRightBtn = document.getElementById('scroll-right-btn');


        // Live statistics elements
        this.liveStats = {
            status: document.getElementById('analysis-status'),
            tokenCount: document.getElementById('live-token-count'),
            charCount: document.getElementById('live-char-count'),
            avgEntropy: document.getElementById('live-avg-entropy'),
            avgVariance: document.getElementById('live-avg-varentropy'),
            avgSkewentropy: document.getElementById('live-avg-skewentropy'),
            avgPerplexity: document.getElementById('live-avg-perplexity'),
            avgProbability: document.getElementById('live-avg-probability'),
            avgLogProb: document.getElementById('live-avg-log-prob')
        };


        this.tooltip = null;
    }

    setupEventListeners() {
        // Textarea events
        this.textarea.addEventListener('input', () => this.handleTextInput());
        this.textarea.addEventListener('scroll', () => this.syncScroll());
        this.textarea.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.textarea.addEventListener('dblclick', (e) => this.handleTextareaDoubleClick(e));

        // Additional scroll synchronization - capture all possible scroll triggers
        this.textarea.addEventListener('wheel', () => this.syncScroll());
        this.textarea.addEventListener('mousewheel', () => this.syncScroll());
        this.textarea.addEventListener('DOMMouseScroll', () => this.syncScroll());
        this.textarea.addEventListener('keyup', () => this.syncScroll());
        this.textarea.addEventListener('keydown', () => this.syncScroll());
        this.textarea.addEventListener('click', () => this.syncScroll());
        this.textarea.addEventListener('focus', () => this.syncScroll());
        this.textarea.addEventListener('touchmove', () => this.syncScroll());
        this.textarea.addEventListener('touchend', () => this.syncScroll());

        // Handle resize events that could affect layout
        window.addEventListener('resize', () => {
            this.syncStyles();
            this.syncScroll();
        });

        // Observe mutations to the textarea that could affect scrolling
        if (window.ResizeObserver) {
            const resizeObserver = new ResizeObserver(() => {
                this.syncStyles();
                this.syncScroll();
            });
            resizeObserver.observe(this.textarea);
        }

        // Optimized continuous synchronization with throttling
        let lastScrollTop = 0;
        let syncScheduled = false;
        setInterval(() => {
            const currentScrollTop = this.textarea.scrollTop;
            if (Math.abs(currentScrollTop - lastScrollTop) > 1 && !syncScheduled) {
                syncScheduled = true;
                requestAnimationFrame(() => {
                    this.syncScroll();
                    lastScrollTop = currentScrollTop;
                    syncScheduled = false;
                });
            }
        }, CONFIG.PERFORMANCE.SCROLL_THROTTLE);

        // Toolbar buttons
        document.getElementById('undoButton').addEventListener('click', () => this.undo());
        document.getElementById('redoButton').addEventListener('click', () => this.redo());
        document.getElementById('clearButton').addEventListener('click', () => this.clearText());


        // Bottom panel tabs
        document.querySelectorAll('.bottom-panel-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.textContent.toLowerCase()));
        });

        // Metrics toggles - implement single selection behavior
        document.querySelectorAll('.metrics-toggle input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    // Uncheck all other metrics to ensure only one is active
                    document.querySelectorAll('.metrics-toggle input[type="checkbox"]').forEach(other => {
                        if (other !== e.target) {
                            other.checked = false;
                        }
                    });
                }
                
                this.invalidateMetricsCache(); // Clear cache when metrics change
                this.updateVisualization();
                this.togglePlotVisibility();
                this.updateLegendVisibility();
            });
        });

        // K-value input for candidates panel - add real-time updates
        if (this.kValueInput) {
            this.kValueInput.addEventListener('change', () => this.updateTopKCandidates());
            this.kValueInput.addEventListener('input', () => this.updateTopKCandidates());
            this.kValueInput.addEventListener('keyup', () => this.updateTopKCandidates());
        }

        // Scroll buttons for token explorer
        if (this.scrollLeftBtn) this.scrollLeftBtn.addEventListener('click', () => this.scrollTokenList('left'));
        if (this.scrollRightBtn) this.scrollRightBtn.addEventListener('click', () => this.scrollTokenList('right'));

        // Listen for scroll events to update scroll button states
        if (this.tokenList && this.tokenList.parentElement) {
            this.tokenList.parentElement.addEventListener('scroll', () => this.updateScrollButtons());
        }

        // Token display interactions
        this.tokenDisplay.addEventListener('click', (e) => this.handleTokenClick(e));
        this.tokenDisplay.addEventListener('dblclick', (e) => this.handleTokenDoubleClick(e));
        this.tokenDisplay.addEventListener('mouseover', (e) => this.handleTokenHover(e));
        this.tokenDisplay.addEventListener('mouseout', () => this.hideTooltip());

        // Resizable bottom panel
        this.setupResizablePanel();
    }

    setupKeyboardShortcuts() {
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                this.undo();
                return false;
            } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                e.preventDefault();
                e.stopPropagation();
                this.redo();
                return false;
            }
        });
    }


    handleTextInput() {
        // Clear previous debounce timer
        if (this.renderDebounceTimer) {
            clearTimeout(this.renderDebounceTimer);
        }

        // Update live statistics immediately (lightweight)
        this.updateLiveStatistics();

        // For real-time analysis, reduce debounce significantly
        const currentText = this.textarea.value;
        const isShortText = currentText.length < CONFIG.ANALYSIS.SHORT_TEXT_LENGTH;
        const renderDelay = isShortText ? CONFIG.PERFORMANCE.RENDER_DEBOUNCE_SHORT : CONFIG.PERFORMANCE.RENDER_DEBOUNCE_NORMAL;

        // Debounce the heavy rendering operations
        this.renderDebounceTimer = setTimeout(() => {
            this.performHeavyRender();
        }, renderDelay);
    }

    performHeavyRender() {
        const currentText = this.textarea.value;

        // Skip if text hasn't changed (prevents unnecessary work)
        if (currentText === this.lastProcessedText) {
            return;
        }

        // Prevent concurrent rendering
        if (this.isRendering) {
            return;
        }

        this.isRendering = true;
        this.lastProcessedText = currentText;

        try {
            this.analyzeText();

            // Ensure scroll alignment after text changes
            requestAnimationFrame(() => {
                this.ensureScrollableHeightAlignment();
                this.syncScroll();
                this.isRendering = false;
            });
        } catch (error) {
            console.error('Rendering error:', error);
            this.isRendering = false;
        }
    }

    handleKeyDown(e) {
        // Handle undo/redo shortcuts at textarea level first
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            this.undo();
            return false;
        } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            this.redo();
            return false;
        }

        // Save state before significant changes
        if (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Delete' ||
            e.key === 'Enter' || (e.ctrlKey && (e.key === 'v' || e.key === 'x'))) {
            this.tokenAnalyzer.saveState(this.textarea);
        }
    }


    syncScroll() {
        // Store the target scroll position
        const targetScrollTop = this.textarea.scrollTop;
        const targetScrollLeft = this.textarea.scrollLeft;

        // Synchronize token display
        if (this.tokenDisplay) {
            // Try direct sync first
            this.tokenDisplay.scrollTop = targetScrollTop;
            this.tokenDisplay.scrollLeft = targetScrollLeft;

            // Verify sync worked, and use requestAnimationFrame as fallback if needed
            const actualScrollTop = this.tokenDisplay.scrollTop;
            if (Math.abs(actualScrollTop - targetScrollTop) > 1) {
                // Fallback to requestAnimationFrame
                requestAnimationFrame(() => {
                    this.tokenDisplay.scrollTop = targetScrollTop;
                    this.tokenDisplay.scrollLeft = targetScrollLeft;

                    // If still failing, try height adjustment
                    if (Math.abs(this.tokenDisplay.scrollTop - targetScrollTop) > 1) {
                        this.ensureScrollableHeightAlignment();

                        // Final retry
                        setTimeout(() => {
                            this.tokenDisplay.scrollTop = targetScrollTop;
                            this.tokenDisplay.scrollLeft = targetScrollLeft;
                        }, 0);
                    }
                });
            }
        }
    }

    analyzeText() {
        const text = this.textarea.value;
        // Enable real-time mode for immediate analysis
        this.tokenAnalyzer.analyzeText(text, true);
        this.updateVisualization();
        this.updateTokenExplorer();
        this.updateDistributionPlots();
        // Update live statistics after tokens are analyzed (important for empty text case)
        this.updateLiveStatistics();
    }

    updateVisualization() {
        this.updateTokenDisplay();
        this.updateTokenExplorer();
    }

    updateTokenDisplay() {
        if (!this.tokenDisplay) return;

        const textContent = this.textarea.value;
        if (!textContent) {
            this.tokenDisplay.innerHTML = '';
            return;
        }

        // Skip expensive operations if tokens haven't changed
        if (this.tokenAnalyzer.tokens.length === 0) {
            return;
        }

        // Cache enabled metrics to avoid repeated DOM queries
        const enabledMetrics = this.getEnabledMetrics();

        // Build overlay by reconstructing the exact textarea content with token spans
        // This ensures perfect alignment between overlay and textarea
        const fragment = document.createDocumentFragment();
        let lastPosition = 0;

        this.tokenAnalyzer.tokens.forEach((token, index) => {
            // Add any gap text between tokens (shouldn't happen but safety check)
            if (token.start > lastPosition) {
                const gapText = textContent.substring(lastPosition, token.start);
                if (gapText) {
                    fragment.appendChild(document.createTextNode(gapText));
                }
            }

            // Create span for the token using the exact text from textarea
            const span = document.createElement('span');
            span.className = 'highlighted-token';
            span.setAttribute('data-token-index', index);

            // Add metric-based classes for visual indicators
            this.addMetricClasses(span, token, enabledMetrics);

            if (token.isWhitespace) {
                span.classList.add('whitespace-token');
            }

            if (index === this.tokenAnalyzer.selectedTokenIndex) {
                span.classList.add('selected');
            }

            // Use the exact text from the textarea at this position
            const actualText = textContent.substring(token.start, token.end);
            span.textContent = actualText;

            fragment.appendChild(span);
            lastPosition = token.end;
        });

        // Add any remaining text after the last token
        if (lastPosition < textContent.length) {
            const remainingText = textContent.substring(lastPosition);
            fragment.appendChild(document.createTextNode(remainingText));
        }

        // Single DOM update
        this.tokenDisplay.textContent = ''; // Clear efficiently
        this.tokenDisplay.appendChild(fragment);

        // Batch style and sync operations
        requestAnimationFrame(() => {
            this.ensureScrollableHeightAlignment();
            this.syncScroll();
        });
    }

    validateAlignment() {
        // Debug helper to check if content matches
        if (DEBUG) {
            const overlayText = this.tokenDisplay.textContent || this.tokenDisplay.innerText;
            const textareaText = this.textarea.value;

            if (overlayText !== textareaText) {
                console.warn('Text content mismatch detected:');
                console.log('Textarea:', JSON.stringify(textareaText));
                console.log('Overlay:', JSON.stringify(overlayText));
                console.log('Tokens:', this.tokenAnalyzer.tokens.map(t => `"${t.text}"`));
            }

            // Check scroll alignment
            const scrollDiff = Math.abs(this.tokenDisplay.scrollTop - this.textarea.scrollTop);
            if (scrollDiff > 1) {
                console.warn(`Scroll misalignment detected: ${scrollDiff}px difference`);
            }

            // Check dimensions and scroll heights
            const textareaRect = this.textarea.getBoundingClientRect();
            const overlayRect = this.tokenDisplay.getBoundingClientRect();
            const heightDiff = Math.abs(textareaRect.height - overlayRect.height);
            const scrollHeightDiff = Math.abs(this.textarea.scrollHeight - this.tokenDisplay.scrollHeight);

            if (heightDiff > 2) {
                console.warn(`Client height mismatch detected: ${heightDiff}px difference`);
                console.log('Textarea height:', textareaRect.height);
                console.log('Overlay height:', overlayRect.height);
            }

            if (scrollHeightDiff > 2) {
                console.warn(`Scroll height mismatch detected: ${scrollHeightDiff}px difference`);

                // Check if overlay is significantly taller (indicating extension issue)
                if (this.tokenDisplay.scrollHeight > this.textarea.scrollHeight + 10) {
                    console.warn('Overlay appears to be extended beyond textarea content - possible tokenization issue');
                }
            }

        }
    }

    syncStyles() {
        if (!this.tokenDisplay || !this.textarea) return;

        // Copy computed styles that affect text layout from textarea to overlay
        const textareaStyles = window.getComputedStyle(this.textarea);
        const overlay = this.tokenDisplay;

        // Copy critical text spacing properties
        overlay.style.fontSize = textareaStyles.fontSize;
        overlay.style.lineHeight = textareaStyles.lineHeight;
        overlay.style.fontFamily = textareaStyles.fontFamily;
        overlay.style.letterSpacing = textareaStyles.letterSpacing;
        overlay.style.wordSpacing = textareaStyles.wordSpacing;
        overlay.style.textIndent = textareaStyles.textIndent;
        // Don't copy padding - we want consistent padding defined in CSS
        overlay.style.borderTopWidth = textareaStyles.borderTopWidth;
        overlay.style.borderRightWidth = textareaStyles.borderRightWidth;
        overlay.style.borderBottomWidth = textareaStyles.borderBottomWidth;
        overlay.style.borderLeftWidth = textareaStyles.borderLeftWidth;

        // Ensure consistent box-sizing and overflow behavior
        overlay.style.boxSizing = textareaStyles.boxSizing;
        overlay.style.overflowWrap = textareaStyles.overflowWrap;
        overlay.style.wordBreak = textareaStyles.wordBreak;
        overlay.style.whiteSpace = 'pre-wrap'; // Ensure consistent whitespace handling

        // Set dimensions to match textarea exactly
        overlay.style.width = textareaStyles.width;
        overlay.style.height = textareaStyles.height;
        overlay.style.minHeight = textareaStyles.minHeight;
        overlay.style.maxHeight = textareaStyles.maxHeight;
    }

    ensureScrollableHeightAlignment() {
        if (!this.tokenDisplay || !this.textarea) return;

        // Clean up any existing adjustments first
        const existingOverlayAdjustments = this.tokenDisplay.querySelectorAll('.height-adjustment, .trailing-newline-spacer');
        existingOverlayAdjustments.forEach(el => el.remove());

        // Force layout calculation after cleanup
        this.textarea.offsetHeight;
        this.tokenDisplay.offsetHeight;

        // Get the actual scrollable heights
        const textareaScrollHeight = this.textarea.scrollHeight;
        const overlayScrollHeight = this.tokenDisplay.scrollHeight;

        if (DEBUG) {
            console.log('Height alignment check:', {
                textarea: textareaScrollHeight,
                overlay: overlayScrollHeight,
                overlayDiff: textareaScrollHeight - overlayScrollHeight
            });
        }

        // Only adjust if there's any height mismatch
        const overlayHeightDiff = textareaScrollHeight - overlayScrollHeight;

        // Adjust overlay height if needed
        if (Math.abs(overlayHeightDiff) > 0) {
            if (overlayHeightDiff > 0) {
                const adjustmentElement = document.createElement('div');
                adjustmentElement.className = 'height-adjustment';
                adjustmentElement.style.cssText = `
                    height: ${overlayHeightDiff}px;
                    width: 0;
                    visibility: hidden;
                    pointer-events: none;
                    overflow: hidden;
                    line-height: 0;
                    font-size: 0;
                `;
                this.tokenDisplay.appendChild(adjustmentElement);

                if (DEBUG) {
                    console.log('Added overlay height adjustment:', overlayHeightDiff, 'px');
                }
            }
        }
    }

    getEnabledMetrics() {
        // Return cached metrics if available
        if (this.cachedMetrics) {
            return this.cachedMetrics;
        }

        // Compute and cache metrics
        this.cachedMetrics = {
            entropy: document.getElementById('entropy-toggle')?.checked || false,
            varentropy: document.getElementById('varentropy-toggle')?.checked || false,
            skewentropy: document.getElementById('skewentropy-toggle')?.checked || false,
            perplexity: document.getElementById('perplexity-toggle')?.checked || false,
            prob: document.getElementById('prob-toggle')?.checked || false,
            logprob: document.getElementById('logprob-toggle')?.checked || false
        };

        return this.cachedMetrics;
    }

    invalidateMetricsCache() {
        this.cachedMetrics = null;
    }

    addMetricClasses(span, token, enabledMetrics) {
        // Find the first enabled metric to display
        const activeMetric = this.getActiveMetric(enabledMetrics);
        
        if (!activeMetric) return;

        // Get the metric value and normalize it
        const { metric, value, normalizedValue } = this.getMetricValue(token, activeMetric);
        
        if (normalizedValue === null) return;

        // Add the metric class with intensity level
        const intensityLevel = this.getIntensityLevel(normalizedValue);
        span.classList.add(`metric-${metric}`, `intensity-${intensityLevel}`);
        
        // Store the actual value for tooltips
        span.setAttribute('data-metric-name', metric);
        span.setAttribute('data-metric-value', this.safeFormatMetric(value, 3));
        span.setAttribute('data-normalized-value', this.safeFormatMetric(normalizedValue, 3));
    }

    getActiveMetric(enabledMetrics) {
        // Return the first enabled metric in priority order
        const metricPriority = ['entropy', 'varentropy', 'skewentropy', 'perplexity', 'prob', 'logprob'];
        
        for (const metric of metricPriority) {
            if (enabledMetrics[metric]) {
                return metric;
            }
        }
        return null;
    }

    getMetricValue(token, metric) {
        let value, normalizedValue;
        
        // Helper function to safely get and validate metric values
        const safeMetricValue = (tokenValue, fallback) => {
            if (tokenValue === null || tokenValue === undefined || isNaN(tokenValue) || !isFinite(tokenValue)) {
                return fallback;
            }
            return tokenValue;
        };
        
        switch (metric) {
            case 'entropy':
                value = safeMetricValue(token.entropy, CONFIG.FALLBACKS.ENTROPY);
                normalizedValue = Math.min(Math.max(value, 0) / CONFIG.NORMALIZATION.ENTROPY_DIVISOR, 1);
                break;
            case 'varentropy':
                value = safeMetricValue(token.varentropy, CONFIG.FALLBACKS.VARENTROPY);
                normalizedValue = Math.min(Math.max(value, 0) / CONFIG.NORMALIZATION.VARENTROPY_DIVISOR, 1);
                break;
            case 'skewentropy':
                value = safeMetricValue(token.skewentropy, CONFIG.FALLBACKS.SKEWENTROPY);
                normalizedValue = Math.min(Math.abs(value) / CONFIG.NORMALIZATION.SKEWENTROPY_DIVISOR, 1);
                break;
            case 'perplexity':
                value = safeMetricValue(token.perplexity, CONFIG.FALLBACKS.PERPLEXITY);
                normalizedValue = Math.min(Math.max(value - CONFIG.NORMALIZATION.PERPLEXITY_OFFSET, 0) / CONFIG.NORMALIZATION.PERPLEXITY_RANGE, 1);
                break;
            case 'prob':
                value = safeMetricValue(token.probability, CONFIG.FALLBACKS.PROBABILITY);
                normalizedValue = Math.min(Math.max(value, 0), 1); // Already in 0-1 range
                break;
            case 'logprob':
                value = safeMetricValue(token.logProbability, CONFIG.FALLBACKS.LOG_PROBABILITY);
                normalizedValue = Math.max(0, Math.min(1, 1 + value / CONFIG.NORMALIZATION.LOG_PROB_SCALE));
                break;
            default:
                return { metric, value: 0, normalizedValue: null };
        }
        
        // Final safety check for normalizedValue
        if (isNaN(normalizedValue) || !isFinite(normalizedValue)) {
            if (DEBUG) {
                console.warn(`Invalid normalized value for metric ${metric}: ${normalizedValue}, using 0`);
            }
            normalizedValue = 0;
        }
        
        return { metric, value, normalizedValue };
    }

    /**
     * Map normalized metric value (0-1) to CSS intensity class
     * @param {number} normalizedValue - Normalized metric value between 0 and 1
     * @returns {string} CSS class name for intensity level
     */
    getIntensityLevel(normalizedValue) {
        // Validate input
        if (normalizedValue === null || normalizedValue === undefined || isNaN(normalizedValue) || !isFinite(normalizedValue)) {
            if (DEBUG) {
                console.warn(`Invalid normalizedValue for intensity: ${normalizedValue}, using very-low`);
            }
            return 'very-low';
        }

        // Clamp to 0-1 range
        const clampedValue = Math.max(0, Math.min(1, normalizedValue));

        // Map normalized value (0-1) to intensity levels using configuration thresholds
        if (clampedValue >= CONFIG.INTENSITY_THRESHOLDS.VERY_HIGH) return 'very-high';
        if (clampedValue >= CONFIG.INTENSITY_THRESHOLDS.HIGH) return 'high';
        if (clampedValue >= CONFIG.INTENSITY_THRESHOLDS.MEDIUM) return 'medium';
        if (clampedValue >= CONFIG.INTENSITY_THRESHOLDS.LOW) return 'low';
        return 'very-low';
    }

    // Safe formatting for metric display
    safeFormatMetric(value, decimals = 3, fallback = '0.000') {
        if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
            return fallback;
        }
        return value.toFixed(decimals);
    }

    updateTokenExplorer() {
        if (!this.tokenList) return;

        const tokens = this.tokenAnalyzer.tokens;

        if (tokens.length === 0) {
            this.tokenList.innerHTML = `
                <div class="no-tokens-message">
                    <p>Start typing to see token analysis</p>
                </div>
            `;
            return;
        }

        // Show all tokens including whitespace
        const displayTokens = tokens;

        // Get enabled metrics and active metric for consistent coloring
        const enabledMetrics = this.getEnabledMetrics();
        const activeMetric = this.getActiveMetric(enabledMetrics);

        // Build token cards HTML
        let explorerHtml = '';
        displayTokens.forEach((token, originalIndex) => {

            // Use the same color coding as the main text area
            let metricClasses = '';
            let tokenTypeClass = '';

            if (token.isWhitespace) {
                tokenTypeClass = 'whitespace';
            } else {
                // Apply the same metric coloring logic
                if (activeMetric) {
                    const { normalizedValue } = this.getMetricValue(token, activeMetric);
                    if (normalizedValue !== null) {
                        const intensityLevel = this.getIntensityLevel(normalizedValue);
                        metricClasses = `metric-${activeMetric} intensity-${intensityLevel}`;
                    }
                }
            }

            // Handle special characters and whitespace for display
            let visibleToken = token.text;
            let tokenLabel = '';

            if (token.isWhitespace) {
                if (token.text === ' ') {
                    visibleToken = CONFIG.TOKENS.SPACE_CHAR;
                    tokenLabel = CONFIG.TOKENS.LABEL_SPACE;
                } else if (token.text === '\t') {
                    visibleToken = CONFIG.TOKENS.TAB_CHAR;
                    tokenLabel = CONFIG.TOKENS.LABEL_TAB;
                } else if (token.text === '\n') {
                    visibleToken = CONFIG.TOKENS.NEWLINE_CHAR;
                    tokenLabel = CONFIG.TOKENS.LABEL_NEWLINE;
                } else if (token.text === '\r\n') {
                    visibleToken = CONFIG.TOKENS.NEWLINE_CHAR;
                    tokenLabel = CONFIG.TOKENS.LABEL_CRLF;
                } else {
                    visibleToken = CONFIG.TOKENS.WHITESPACE_CHAR;
                    tokenLabel = CONFIG.TOKENS.LABEL_WHITESPACE;
                }
            } else {
                // Escape HTML for regular tokens
                visibleToken = visibleToken.replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#39;');
            }

            // Get metric classes
            const getMetricClass = (value) => {
                if (value > 0.7) return 'high';
                if (value > 0.4) return 'medium';
                return 'low';
            };

            const isSelected = originalIndex === this.tokenAnalyzer.selectedTokenIndex;

            // Build metrics HTML - show all metrics
            let metricsHtml = '';
            const allMetrics = [
                { key: 'entropy', label: 'Entropy', value: token.entropy },
                { key: 'varentropy', label: 'Varentropy', value: token.varentropy },
                { key: 'skewentropy', label: 'Skewentropy', value: token.skewentropy },
                { key: 'perplexity', label: 'Perplexity', value: token.perplexity },
                { key: 'prob', label: 'Prob.', value: token.probability },
                { key: 'logprob', label: 'Log Prob.', value: token.logProbability }
            ];

            // Display all metrics
            allMetrics.forEach(metric => {
                const precision = metric.key === 'perplexity' ? 1 : 3;
                const isActive = metric.key === activeMetric ? 'active-metric' : '';
                
                metricsHtml += `
                    <div class="explorer-metric ${isActive}">
                        <span class="explorer-metric-label">${metric.label}:</span>
                        <span class="explorer-metric-value">${this.safeFormatMetric(metric.value, precision)}</span>
                    </div>
                `;
            });

            explorerHtml += `
                <div class="explorer-token ${tokenTypeClass} ${metricClasses} ${isSelected ? 'selected' : ''}" data-index="${originalIndex}">
                    <div class="explorer-token-text">${visibleToken}</div>
                    <div class="explorer-token-metrics">
                        ${metricsHtml}
                    </div>
                </div>
            `;
        });

        this.tokenList.innerHTML = explorerHtml;

        // Add click handlers to token cards
        this.tokenList.querySelectorAll('.explorer-token').forEach(card => {
            const index = parseInt(card.getAttribute('data-index'));
            card.addEventListener('click', () => this.selectExplorerToken(index));
            card.addEventListener('dblclick', () => this.jumpToToken(index));
        });

        // Update scroll buttons after content change
        setTimeout(() => this.updateScrollButtons(), 0);
    }

    createTokenCard(token, index) {
        const card = document.createElement('div');
        card.className = 'token-card';
        card.dataset.index = index;

        if (index === this.tokenAnalyzer.selectedTokenIndex) {
            card.classList.add('selected');
        }

        card.innerHTML = `
            <div class="token-header">
                <div class="token-text">${this.escapeHtml(token.text)}</div>
                <div class="token-index">#${index}</div>
            </div>
            <div class="token-metrics">
                <div class="metric-row">
                    <span class="metric-label">Probability</span>
                    <span class="metric-value ${this.getMetricClass(token.probability)}">${token.probability.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Log Prob</span>
                    <span class="metric-value">${token.logProbability.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Entropy</span>
                    <span class="metric-value">${token.entropy.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Varentropy</span>
                    <span class="metric-value">${token.varentropy.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Perplexity</span>
                    <span class="metric-value">${token.perplexity.toFixed(1)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Attention</span>
                    <span class="metric-value ${this.getMetricClass(token.attention)}">${token.attention.toFixed(3)}</span>
                </div>
            </div>
        `;

        card.addEventListener('click', () => this.selectToken(index));
        card.addEventListener('dblclick', () => this.jumpToToken(index));

        return card;
    }

    getMetricClass(value) {
        if (value > 0.7) return 'high';
        if (value > 0.4) return 'medium';
        return 'low';
    }

    selectToken(index) {
        this.tokenAnalyzer.selectToken(index);
        this.updateTokenDisplay();
        this.updateTokenExplorer();
        this.updateTopKCandidates();
    }

    jumpToToken(index) {
        if (DEBUG) {
            console.log('Jump to Token:', index);
        }
        const token = this.tokenAnalyzer.tokens[index];
        if (token) {
            this.textarea.focus();
            this.textarea.setSelectionRange(token.start, token.end);
        }
    }

    updateTopKCandidates() {
        if (!this.candidatesList || !this.candidatePosition) {
            return;
        }

        const selectedIndex = this.tokenAnalyzer.selectedTokenIndex;
        const token = this.tokenAnalyzer.tokens[selectedIndex];

        if (!token || selectedIndex < 0) {
            this.candidatesList.innerHTML = `
                <div class="no-selection-message">
                    <p>Select a token to view top-k candidates</p>
                </div>
            `;
            this.candidatePosition.textContent = '-';
            return;
        }

        // Update position display
        this.candidatePosition.textContent = selectedIndex;

        // Get k value with validation
        let k = parseInt(this.kValueInput?.value) || CONFIG.ANALYSIS.DEFAULT_TOP_K;
        // Ensure k is within valid range
        k = Math.max(1, Math.min(k, CONFIG.ANALYSIS.MAX_TOP_K));

        // Update the input field if the value was clamped
        if (this.kValueInput && parseInt(this.kValueInput.value) !== k) {
            this.kValueInput.value = k;
        }

        // Get candidates
        const candidates = this.tokenAnalyzer.getTopKCandidates(selectedIndex, k);

        // Build candidates HTML
        let candidatesHtml = '';
        if (candidates && candidates.length > 0) {
            candidates.forEach((candidate, index) => {
                const isSelected = candidate.token === token.text;
                candidatesHtml += `
                    <div class="candidate-item ${isSelected ? 'selected-candidate' : ''}" data-rank="${index + 1}">
                        <span class="candidate-token">${this.formatTokenForDisplay(candidate.token)}</span>
                        <span class="candidate-prob ${isSelected ? 'selected-prob' : ''}">${(candidate.probability * 100).toFixed(1)}%</span>
                    </div>
                `;
            });
        } else {
            candidatesHtml = '<div class="no-selection-message"><p>No candidates generated</p></div>';
        }

        this.candidatesList.innerHTML = candidatesHtml;

        // Add click handlers for candidate items
        this.candidatesList.querySelectorAll('.candidate-item').forEach(item => {
            item.addEventListener('click', () => {
                const rank = item.getAttribute('data-rank');
                const candidateToken = item.querySelector('.candidate-token').textContent;
                // Could implement candidate selection functionality here
            });
        });
    }

    selectExplorerToken(index) {
        // Clear previous selections in token list
        if (this.tokenList) {
            this.tokenList.querySelectorAll('.explorer-token.selected').forEach(el => {
                el.classList.remove('selected');
            });

            // Select the clicked token
            const explorerToken = this.tokenList.querySelector(`[data-index="${index}"]`);
            if (explorerToken) {
                explorerToken.classList.add('selected');

                // Scroll the selected token into view
                this.scrollTokenIntoView(explorerToken);
            }
        }

        // Update the main token selection
        this.selectToken(index);
    }

    /**
     * Scroll the token list horizontally
     * @param {string} direction - 'left' or 'right'
     */
    scrollTokenList(direction) {
        const container = this.tokenList?.parentElement;
        if (!container) return;

        const tokenWidth = CONFIG.DIMENSIONS.TOKEN_CARD_WIDTH;
        const gap = CONFIG.DIMENSIONS.TOKEN_CARD_GAP;
        const scrollAmount = (tokenWidth + gap) * CONFIG.INTERACTION.TOKENS_PER_SCROLL;

        if (direction === 'left') {
            container.scrollTo({
                left: Math.max(0, container.scrollLeft - scrollAmount),
                behavior: 'smooth'
            });
        } else {
            container.scrollTo({
                left: container.scrollLeft + scrollAmount,
                behavior: 'smooth'
            });
        }
    }

    scrollTokenIntoView(tokenElement) {
        const container = this.tokenList?.parentElement;
        if (!container || !tokenElement) return;

        // Always scroll to center the token for better visibility
        const tokenOffsetLeft = tokenElement.offsetLeft;
        const containerWidth = container.clientWidth;
        const tokenWidth = tokenElement.offsetWidth;

        // Calculate scroll position to center the token
        const targetScrollLeft = tokenOffsetLeft - (containerWidth - tokenWidth) / 2;

        container.scrollTo({
            left: Math.max(0, targetScrollLeft),
            behavior: 'smooth'
        });

        // Update scroll button states after scrolling
        setTimeout(() => this.updateScrollButtons(), CONFIG.INTERACTION.SCROLL_BUTTON_UPDATE_DELAY);
    }

    updateScrollButtons() {
        const container = this.tokenList?.parentElement;
        if (!container || !this.scrollLeftBtn || !this.scrollRightBtn) return;

        const scrollLeft = container.scrollLeft;
        const maxScroll = container.scrollWidth - container.clientWidth;

        this.scrollLeftBtn.disabled = scrollLeft <= 0;
        this.scrollRightBtn.disabled = scrollLeft >= maxScroll - 1;
    }

    jumpToExplorerToken(index) {
        // First select the token in the explorer
        this.selectExplorerToken(index);

        // Switch to the explorer tab if not already active
        this.switchTab('explorer');

        // Scroll to the bottom panel to make the explorer visible
        const bottomPanel = document.getElementById('bottomPanel');
        if (bottomPanel) {
            bottomPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    updateLiveStatistics() {
        const stats = this.tokenAnalyzer.getStatistics();

        // Helper function to safely format numeric values
        const safeFormat = (value, decimals = 3, fallback = '0.000') => {
            if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
                return fallback;
            }
            return value.toFixed(decimals);
        };

        // Update basic counts immediately for real-time feel
        if (this.liveStats.tokenCount) this.liveStats.tokenCount.textContent = stats.totalTokens || 0;
        if (this.liveStats.charCount) this.liveStats.charCount.textContent = stats.characters || 0;
        
        // Update metrics with smooth transitions and safe formatting
        if (this.liveStats.avgEntropy) {
            const value = safeFormat(stats.avgEntropy, 3, '0.000');
            if (this.liveStats.avgEntropy.textContent !== value) {
                this.liveStats.avgEntropy.textContent = value;
            }
        }
        if (this.liveStats.avgVariance) {
            const value = safeFormat(stats.avgVariance, 3, '0.000');
            if (this.liveStats.avgVariance.textContent !== value) {
                this.liveStats.avgVariance.textContent = value;
            }
        }
        if (this.liveStats.avgSkewentropy) {
            const value = safeFormat(stats.avgSkewentropy, 3, '0.000');
            if (this.liveStats.avgSkewentropy.textContent !== value) {
                this.liveStats.avgSkewentropy.textContent = value;
            }
        }
        if (this.liveStats.avgPerplexity) {
            const value = safeFormat(stats.avgPerplexity, 3, '1.000');
            if (this.liveStats.avgPerplexity.textContent !== value) {
                this.liveStats.avgPerplexity.textContent = value;
            }
        }
        if (this.liveStats.avgProbability) {
            const value = safeFormat(stats.avgProbability, 3, '0.000');
            if (this.liveStats.avgProbability.textContent !== value) {
                this.liveStats.avgProbability.textContent = value;
            }
        }
        if (this.liveStats.avgLogProb) {
            const value = safeFormat(stats.avgLogProb, 3, '-10.000');
            if (this.liveStats.avgLogProb.textContent !== value) {
                this.liveStats.avgLogProb.textContent = value;
            }
        }
    }

    switchTab(tabName) {
        this.activeTab = tabName;

        // Update tab appearance
        document.querySelectorAll('.bottom-panel-tab').forEach(tab => {
            tab.classList.toggle('active', tab.textContent.toLowerCase() === tabName);
        });

        // Hide all content first
        this.tokenExplorer.style.display = 'none';
        this.distributionsContent.style.display = 'none';
        this.tokenExplorer.classList.remove('active');
        this.distributionsContent.classList.remove('active');

        // Show only the selected content
        if (tabName === 'explorer') {
            this.tokenExplorer.style.display = 'block';
            this.tokenExplorer.classList.add('active');
        } else if (tabName === 'distributions') {
            this.distributionsContent.style.display = 'block';
            this.distributionsContent.classList.add('active');
        }
    }

    undo() {
        if (this.tokenAnalyzer.undo(this.textarea)) {
            this.updateVisualization();
            this.updateLiveStatistics();
        }
    }

    redo() {
        if (this.tokenAnalyzer.redo(this.textarea)) {
            this.updateVisualization();
            this.updateLiveStatistics();
        }
    }

    clearText() {
        this.tokenAnalyzer.saveState(this.textarea);
        this.textarea.value = '';
        this.textarea.focus();
        this.handleTextInput();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format a token for display by making special characters visible
     * @param {string} token - The token to format
     * @returns {string} HTML-escaped token with visible special characters
     */
    formatTokenForDisplay(token) {
        // First escape HTML
        let escaped = this.escapeHtml(token);

        // Replace special characters with visible representations
        escaped = escaped
            .replace(/\n/g, '<span class="special-char" title="Newline">↵</span>')
            .replace(/\t/g, '<span class="special-char" title="Tab">⇥</span>')
            .replace(/\r/g, '<span class="special-char" title="Carriage Return">⏎</span>');

        // Handle leading/trailing spaces with visible marker
        if (escaped.startsWith(' ')) {
            escaped = '<span class="special-char" title="Space">␣</span>' + escaped.substring(1);
        }

        return escaped;
    }


    handleTokenClick(e) {
        const tokenSpan = e.target.closest('[data-token-index]');
        if (tokenSpan) {
            const index = parseInt(tokenSpan.getAttribute('data-token-index'));
            this.selectToken(index);
        }
    }

    handleTokenDoubleClick(e) {
        const tokenSpan = e.target.closest('[data-token-index]');
        if (tokenSpan) {
            const index = parseInt(tokenSpan.getAttribute('data-token-index'));
            this.jumpToExplorerToken(index);
        }
    }

    handleTextareaDoubleClick(e) {
        e.preventDefault(); // Prevent default text selection

        const textarea = this.textarea;
        const text = textarea.value;

        if (!text || text.length === 0) return;

        // Use the same cursor position that the textarea naturally provides
        const cursorPosition = textarea.selectionStart;
        if (DEBUG) {
            console.log('Cursor position:', cursorPosition);
        }

        // Find which token contains the cursor position using the same approach as updateTokenDisplay()
        // This mirrors exactly how the overlay spans are built
        const tokens = this.tokenAnalyzer.tokens;
        let tokenIndex = -1;
        
        // Direct mapping approach: check each token's start/end positions
        for (let i = 0; i < tokens.length; i++) {
            const token = tokens[i];
            
            // Check if cursor is within this token's range (inclusive start, exclusive end)
            if (cursorPosition >= token.start && cursorPosition < token.end) {
                tokenIndex = i;
                if (DEBUG) {
                    console.log(`Found exact match: token ${i} "${token.text}" at position ${token.start}-${token.end}`);
                }
                break;
            }

            // Special case: if cursor is at the very end of a token (and it's the last character)
            if (cursorPosition === token.end && cursorPosition === text.length) {
                tokenIndex = i;
                if (DEBUG) {
                    console.log(`Found end-of-text match: token ${i} "${token.text}" at position ${token.start}-${token.end}`);
                }
                break;
            }
        }

        // If no exact match, find the closest token
        if (tokenIndex === -1) {
            let bestMatch = -1;
            let smallestDistance = Infinity;
            
            for (let i = 0; i < tokens.length; i++) {
                const token = tokens[i];
                const distance = Math.min(
                    Math.abs(cursorPosition - token.start),
                    Math.abs(cursorPosition - token.end)
                );
                
                if (distance < smallestDistance) {
                    smallestDistance = distance;
                    bestMatch = i;
                }
            }
            
            // Use best match if distance is reasonable (within 3 characters)
            if (bestMatch !== -1 && smallestDistance <= 3) {
                tokenIndex = bestMatch;
                if (DEBUG) {
                    console.log(`Using closest match: token ${tokenIndex} with distance ${smallestDistance}`);
                }
            }
        }

        // If we found a valid token, jump to it in the explorer
        if (tokenIndex !== -1) {
            const token = tokens[tokenIndex];
            if (DEBUG) {
                console.log(`Double-clicked at position ${cursorPosition}, mapped to token ${tokenIndex}: "${token.text}" (${token.start}-${token.end})`);
            }
            this.jumpToExplorerToken(tokenIndex);
        } else {
            if (DEBUG) {
                console.log(`Double-clicked at position ${cursorPosition}, no token found.`);
                console.log('Available tokens:', tokens.map((t, i) =>
                    `${i}: "${t.text}" (${t.start}-${t.end})`));
            }
            
            // Verify text reconstruction
            let reconstructed = '';
            let lastPos = 0;
            tokens.forEach(token => {
                if (token.start > lastPos) {
                    reconstructed += text.substring(lastPos, token.start);
                }
                reconstructed += text.substring(token.start, token.end);
                lastPos = token.end;
            });
            if (lastPos < text.length) {
                reconstructed += text.substring(lastPos);
            }

            if (DEBUG) {
                console.log('Text reconstruction matches:', text === reconstructed);
            }
        }
    }

    handleTokenHover(e) {
        const tokenSpan = e.target.closest('[data-token-index]');
        if (tokenSpan) {
            const index = parseInt(tokenSpan.getAttribute('data-token-index'));
            const token = this.tokenAnalyzer.tokens[index];
            if (token) {
                this.showTooltip(e, token);
            }
        }
    }

    showTooltip(e, token) {
        this.hideTooltip();

        this.tooltip = document.createElement('div');
        this.tooltip.className = 'token-tooltip';

        // Handle different token types in tooltip
        let tokenDisplay = token.text;
        let tokenType = 'Token';

        if (token.isWhitespace) {
            if (token.text === ' ') {
                tokenDisplay = '[SPACE]';
                tokenType = 'Space';
            } else if (token.text === '\n') {
                tokenDisplay = '[NEWLINE]';
                tokenType = 'Newline';
            } else if (token.text === '\t') {
                tokenDisplay = '[TAB]';
                tokenType = 'Tab';
            } else if (token.text === '\r\n') {
                tokenDisplay = '[CRLF]';
                tokenType = 'Line Break';
            } else {
                tokenDisplay = '[WHITESPACE]';
                tokenType = 'Whitespace';
            }
        } else if (token.isPunctuation) {
            tokenType = 'Punctuation';
        } else if (token.isWord) {
            tokenType = 'Word';
        }

        // Build tooltip content with active metric focus
        const enabledMetrics = this.getEnabledMetrics();
        const activeMetric = this.getActiveMetric(enabledMetrics);
        
        let tooltipContent = `<div><strong>${this.escapeHtml(tokenDisplay)}</strong> (${tokenType})</div>`;
        
        // Show raw tokenizer string if different from decoded text
        if (token.rawText && token.rawText !== token.text) {
            tooltipContent += `<div><small>Raw: ${this.escapeHtml(token.rawText)}</small></div>`;
        }
        
        // Show the active metric prominently
        if (activeMetric) {
            const metricNames = {
                entropy: 'Entropy',
                varentropy: 'Varentropy', 
                skewentropy: 'Skewentropy',
                perplexity: 'Perplexity',
                prob: 'Probability',
                logprob: 'Log Probability'
            };
            
            const { value } = this.getMetricValue(token, activeMetric);
            tooltipContent += `<div><strong>${metricNames[activeMetric]}: ${this.safeFormatMetric(value, 3)}</strong></div>`;
        }
        
        // Always show some key metrics for context
        tooltipContent += `<div>Position: ${token.start}-${token.end}</div>`;
        
        this.tooltip.innerHTML = tooltipContent;

        document.body.appendChild(this.tooltip);

        const rect = e.target.getBoundingClientRect();
        this.tooltip.style.left = `${rect.left + rect.width / 2}px`;
        this.tooltip.style.top = `${rect.top - this.tooltip.offsetHeight - 5}px`;

        // Adjust if tooltip goes off screen
        const tooltipRect = this.tooltip.getBoundingClientRect();
        if (tooltipRect.left < 5) {
            this.tooltip.style.left = '5px';
        } else if (tooltipRect.right > window.innerWidth - 5) {
            this.tooltip.style.left = `${window.innerWidth - tooltipRect.width - 5}px`;
        }
    }

    hideTooltip() {
        if (this.tooltip) {
            this.tooltip.remove();
            this.tooltip = null;
        }
    }

    setupResizablePanel() {
        const resizeHandle = document.getElementById('resizeHandle');
        const bottomPanel = document.getElementById('bottomPanel');

        if (!resizeHandle || !bottomPanel) {
            if (DEBUG) {
                console.log('Resize elements not found:', { resizeHandle, bottomPanel });
            }
            return;
        }

        let isResizing = false;
        let startY = 0;
        let startHeight = 0;

        // Make sure resize handle is clickable and visible
        resizeHandle.style.cursor = 'row-resize';
        resizeHandle.style.zIndex = '1000';
        resizeHandle.title = 'Drag to resize panel';

        const startResize = (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = parseInt(window.getComputedStyle(bottomPanel).height, 10);

            document.body.style.cursor = 'row-resize';
            document.body.style.userSelect = 'none';
            document.body.style.pointerEvents = 'none'; // Prevent interference
            resizeHandle.style.pointerEvents = 'auto'; // Keep handle interactive

            // Add visual feedback
            bottomPanel.style.transition = 'none'; // Disable transition during resize
            resizeHandle.style.backgroundColor = '#0353e9';

            e.preventDefault();
            e.stopPropagation();
        };

        const doResize = (e) => {
            if (!isResizing) return;

            const currentY = e.clientY;
            const deltaY = startY - currentY; // Inverted because we're resizing from top
            const newHeight = Math.max(50, Math.min(window.innerHeight * 0.9, startHeight + deltaY));

            bottomPanel.style.height = `${newHeight}px`;

            e.preventDefault();
            e.stopPropagation();
        };

        const stopResize = () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                document.body.style.pointerEvents = '';
                resizeHandle.style.pointerEvents = '';

                // Restore visual state
                bottomPanel.style.transition = '';
                resizeHandle.style.backgroundColor = '';
            }
        };

        // Event listeners
        resizeHandle.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', doResize);
        document.addEventListener('mouseup', stopResize);

        // Fallback for when mouse leaves window
        document.addEventListener('mouseleave', stopResize);
    }

    // Plotting Methods
    initializePlots() {
        this.plotCanvases = {
            entropy: document.querySelector('#entropy-plot canvas'),
            varentropy: document.querySelector('#varentropy-plot canvas'),
            skewentropy: document.querySelector('#skewentropy-plot canvas'),
            perplexity: document.querySelector('#perplexity-plot canvas'),
            probability: document.querySelector('#probability-plot canvas'),
            logprob: document.querySelector('#logprob-plot canvas'),
            scatter: document.querySelector('#entropy-varentropy-scatter canvas')
        };

        // Initialize plot visibility based on metrics toggles
        this.togglePlotVisibility();
    }

    updateDistributionPlots() {
        if (!this.tokenAnalyzer.tokens || this.tokenAnalyzer.tokens.length === 0) {
            return;
        }

        const tokens = this.tokenAnalyzer.tokens.filter(t => !t.isWhitespace);
        if (tokens.length === 0) return;

        // Draw histograms for all metrics (always show all in distributions tab)
        if (this.plotCanvases.entropy) {
            this.drawHistogram(this.plotCanvases.entropy, tokens.map(t => t.entropy), 'Entropy', '#0f62fe');
        }
        if (this.plotCanvases.varentropy) {
            this.drawHistogram(this.plotCanvases.varentropy, tokens.map(t => t.varentropy), 'Varentropy', '#24a148');
        }
        if (this.plotCanvases.skewentropy) {
            this.drawHistogram(this.plotCanvases.skewentropy, tokens.map(t => t.skewentropy), 'Skewentropy', '#f1c21b');
        }
        if (this.plotCanvases.perplexity) {
            this.drawHistogram(this.plotCanvases.perplexity, tokens.map(t => t.perplexity), 'Perplexity', '#da1e28');
        }
        if (this.plotCanvases.probability) {
            this.drawHistogram(this.plotCanvases.probability, tokens.map(t => t.probability), 'Probability', '#8a3ffc');
        }
        if (this.plotCanvases.logprob) {
            this.drawHistogram(this.plotCanvases.logprob, tokens.map(t => t.logProbability), 'Log Probability', '#fa4d56');
        }

        // Always draw scatter plot in distributions tab
        if (this.plotCanvases.scatter) {
            this.drawScatterPlot(this.plotCanvases.scatter, tokens);
        }
    }

    drawHistogram(canvas, data, label, color) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (data.length === 0) return;

        // Calculate histogram bins
        const min = Math.min(...data);
        const max = Math.max(...data);
        const binCount = Math.min(20, Math.max(5, Math.floor(Math.sqrt(data.length))));
        const binWidth = (max - min) / binCount;

        const bins = new Array(binCount).fill(0);
        data.forEach(value => {
            const binIndex = Math.min(binCount - 1, Math.floor((value - min) / binWidth));
            bins[binIndex]++;
        });

        const maxBinCount = Math.max(...bins);

        // Set up drawing parameters (scaled for higher resolution)
        const padding = 80;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        const barWidth = chartWidth / binCount;

        // Draw bars
        ctx.fillStyle = color;
        bins.forEach((count, i) => {
            const barHeight = (count / maxBinCount) * chartHeight;
            const x = padding + i * barWidth;
            const y = height - padding - barHeight;

            ctx.fillRect(x, y, barWidth * 0.8, barHeight);
        });

        // Draw axes
        ctx.strokeStyle = '#525252';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();

        // Draw labels (scaled font size)
        ctx.fillStyle = '#161616';
        ctx.font = '24px IBM Plex Sans';
        ctx.textAlign = 'center';

        // X-axis labels
        for (let i = 0; i <= binCount; i += Math.ceil(binCount / 5)) {
            const x = padding + i * barWidth;
            const value = min + i * binWidth;
            ctx.fillText(value.toFixed(2), x, height - padding + 30);
        }

        // Y-axis labels
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = height - padding - (i / 5) * chartHeight;
            const value = Math.round((i / 5) * maxBinCount);
            ctx.fillText(value.toString(), padding - 10, y + 8);
        }
    }

    drawScatterPlot(canvas, tokens) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (tokens.length === 0) return;

        const entropyData = tokens.map(t => t.entropy);
        const varentropyData = tokens.map(t => t.varentropy);

        // Always start from 0,0 and extend to data max
        const entropyMin = 0;
        const entropyMax = Math.max(...entropyData);
        const varentropyMin = 0;
        const varentropyMax = Math.max(...varentropyData);

        // Set up drawing parameters (scaled for higher resolution)
        const padding = 100;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;

        // Draw points (larger for higher resolution)
        ctx.fillStyle = '#0f62fe';
        tokens.forEach(token => {
            const x = padding + ((token.entropy - entropyMin) / (entropyMax - entropyMin)) * chartWidth;
            const y = height - padding - ((token.varentropy - varentropyMin) / (varentropyMax - varentropyMin)) * chartHeight;

            ctx.beginPath();
            ctx.arc(x, y, 6, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Draw axes (thicker for higher resolution)
        ctx.strokeStyle = '#525252';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();

        // Draw axis labels (scaled font size)
        ctx.fillStyle = '#161616';
        ctx.font = '24px IBM Plex Sans';

        // X-axis labels (Entropy) - starting from 0
        ctx.textAlign = 'center';
        for (let i = 0; i <= 5; i++) {
            const x = padding + (i / 5) * chartWidth;
            const value = entropyMin + (i / 5) * (entropyMax - entropyMin);
            ctx.fillText(value.toFixed(1), x, height - padding + 30);
        }

        // Y-axis labels (Varentropy) - starting from 0
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = height - padding - (i / 5) * chartHeight;
            const value = varentropyMin + (i / 5) * (varentropyMax - varentropyMin);
            ctx.fillText(value.toFixed(1), padding - 10, y + 8);
        }

        // Axis titles (scaled font size)
        ctx.textAlign = 'center';
        ctx.font = '28px IBM Plex Sans';
        ctx.fillText('Entropy', width / 2, height - 20);

        ctx.save();
        ctx.translate(30, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Varentropy', 0, 0);
        ctx.restore();
    }

    togglePlotVisibility() {
        // In the distributions tab, show all plots regardless of toggle state
        const plotElements = {
            entropy: document.getElementById('entropy-plot'),
            varentropy: document.getElementById('varentropy-plot'),
            skewentropy: document.getElementById('skewentropy-plot'),
            perplexity: document.getElementById('perplexity-plot'),
            prob: document.getElementById('probability-plot'),
            logprob: document.getElementById('logprob-plot')
        };

        // Always show all metric plots in distributions tab
        Object.keys(plotElements).forEach(metric => {
            const element = plotElements[metric];
            if (element) {
                element.style.display = 'block';
            }
        });

        // Always show scatter plot in distributions tab
        const scatterPlot = document.getElementById('entropy-varentropy-scatter');
        if (scatterPlot) {
            scatterPlot.style.display = 'block';
        }

        // Update all plots
        this.updateDistributionPlots();
    }

    updateLegendVisibility() {
        const enabledMetrics = this.getEnabledMetrics();
        const activeMetric = this.getActiveMetric(enabledMetrics);
        
        // Hide all legend items first
        document.querySelectorAll('.legend-item').forEach(item => {
            item.style.display = 'none';
        });
        
        // Show only the active metric's legend
        if (activeMetric) {
            const activeLegendItem = document.querySelector(`[data-metric="${activeMetric}"]`);
            if (activeLegendItem) {
                activeLegendItem.style.display = 'flex';
            }
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    window.entroscope = new LogitScope();
});