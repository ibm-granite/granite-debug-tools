/**
 * LogitScope UI Configuration
 *
 * Central configuration file for all UI constants and settings.
 * This file contains all magic numbers, color values, and configuration
 * parameters used throughout the LogitScope web interface.
 *
 * @module config
 */

export const CONFIG = {
  /**
   * Debug and Development
   * Set to false in production to disable debug logging and validation
   */
  DEBUG: false,

  /**
   * UI Dimensions (pixels)
   */
  DIMENSIONS: {
    SIDEBAR_WIDTH: 320,
    TOKEN_CARD_WIDTH: 180,
    TOKEN_CARD_HEIGHT: 240,
    MIN_PANEL_HEIGHT: 50,
    TOKEN_CARD_GAP: 8, // Gap between token cards in explorer
  },

  /**
   * Analysis and Performance Settings
   */
  ANALYSIS: {
    MAX_UNDO_STEPS: 50,
    DEBOUNCE_DELAY: 50, // milliseconds - base delay for analysis
    DEFAULT_TOP_K: 10, // Default number of top-k candidates to show
    MAX_TOP_K: 50, // Maximum allowed top-k value

    // Adaptive debounce delays based on input characteristics
    DEBOUNCE_DELAY_SHORT: 10, // For very short text (< 10 chars)
    DEBOUNCE_DELAY_FAST: 25, // For short sentences (< 50 chars)
    DEBOUNCE_DELAY_NORMAL: 50, // Standard delay

    // Text length thresholds for adaptive debouncing
    SHORT_TEXT_LENGTH: 10,
    FAST_TEXT_LENGTH: 50,
  },

  /**
   * Metric Clamping Values
   * Used by the backend and frontend for handling extreme values
   */
  METRICS: {
    ENTROPY_MAX: 20.0,
    VARENTROPY_MAX: 10.0,
    SKEWENTROPY_MAX: 10.0,
    PERPLEXITY_MAX: 1000.0,
    PERPLEXITY_MIN: 1.0,
    SURPRISAL_MAX: 50.0,
    LOG_PROB_MIN: -50.0,
    LOG_PROB_MAX: 0.0,
    PROBABILITY_MIN: 0.0,
    PROBABILITY_MAX: 1.0,
    DEFAULT_MAX: 100.0,
  },

  /**
   * Metric Normalization for Visualization
   * Maps raw metric values to 0-1 range for color intensity
   */
  NORMALIZATION: {
    // Entropy: typical range 0-4
    ENTROPY_DIVISOR: 4,

    // Varentropy: typical range 0-2
    VARENTROPY_DIVISOR: 2,

    // Skewentropy: use absolute value, typical range 0-2
    SKEWENTROPY_DIVISOR: 2,

    // Perplexity: typical range 1-20, normalized to 0-1
    PERPLEXITY_OFFSET: 1,
    PERPLEXITY_RANGE: 19,

    // Log probability: typical range -10 to 0, normalized to 0-1
    LOG_PROB_SCALE: 10,
  },

  /**
   * Intensity Level Thresholds
   * Maps normalized values (0-1) to CSS intensity classes
   */
  INTENSITY_THRESHOLDS: {
    VERY_HIGH: 0.8,
    HIGH: 0.6,
    MEDIUM: 0.4,
    LOW: 0.2,
    // Below 0.2 = very-low
  },

  /**
   * Color Palette
   * Based on IBM Carbon Design System
   */
  COLORS: {
    // Primary colors
    PRIMARY: '#0f62fe',
    PRIMARY_HOVER: '#0353e9',

    // Background colors
    BACKGROUND_DARK: '#161616',
    SURFACE_01: '#262626',
    SURFACE_02: '#393939',
    SURFACE_03: '#525252',

    // Text colors
    TEXT_PRIMARY: '#f4f4f4',
    TEXT_SECONDARY: '#c6c6c6',
    TEXT_DISABLED: '#6f6f6f',

    // Border colors
    BORDER: '#525252',
    BORDER_SUBTLE: '#393939',

    // Metric-specific colors
    METRIC_ENTROPY: '#0f62fe',
    METRIC_VARENTROPY: '#24a148',
    METRIC_SKEWENTROPY: '#f1c21b',
    METRIC_PERPLEXITY: '#da1e28',
    METRIC_PROBABILITY: '#8a3ffc',
    METRIC_LOG_PROB: '#fa4d56',

    // Status colors
    STATUS_CONNECTED: '#24a148',
    STATUS_DISCONNECTED: '#da1e28',
    STATUS_ERROR: '#fa4d56',
    STATUS_ANALYZING: '#0f62fe',
  },

  /**
   * Performance and Optimization
   */
  PERFORMANCE: {
    SCROLL_THROTTLE: 32, // milliseconds (30fps for scroll sync)
    RENDER_DEBOUNCE_SHORT: 10, // For very short text
    RENDER_DEBOUNCE_NORMAL: 50, // For normal text
  },

  /**
   * WebSocket Configuration
   */
  WEBSOCKET: {
    RECONNECT_DELAY: 2000, // milliseconds - delay before reconnecting
    PING_MESSAGE: { type: 'ping' },
  },

  /**
   * Token Visualization
   */
  TOKENS: {
    // Special character representations
    SPACE_CHAR: '␣',
    TAB_CHAR: '␉',
    NEWLINE_CHAR: '↵',
    WHITESPACE_CHAR: '⎵',

    // Token type labels
    LABEL_SPACE: 'SPACE',
    LABEL_TAB: 'TAB',
    LABEL_NEWLINE: 'NEWLINE',
    LABEL_CRLF: 'CRLF',
    LABEL_WHITESPACE: 'WHITESPACE',
  },

  /**
   * Tokenizer Decoding
   * Mappings for common tokenizer artifacts
   */
  TOKENIZER: {
    // GPT-style space prefix
    GPT_SPACE_PREFIX: 'Ġ',

    // T5/BERT-style space prefix
    T5_SPACE_PREFIX: '▁',

    // GPT-style newline character
    GPT_NEWLINE_CHAR: 'Ċ',

    // UTF-8 encoding artifacts from BPE tokenizers
    BPE_ARTIFACTS: {
      'âĢĻ': '—',  // em dash
      'âĢķ': '–',  // en dash
      'âĢĿ': '"',  // left double quote
      'âĢĺ': '"',  // right double quote
      'âĢļ': '\'', // left single quote
      'âĢĹ': '\'', // right single quote
      'âĨ': '…',   // ellipsis
    },
  },

  /**
   * Plot Configuration
   */
  PLOTS: {
    // Histogram settings
    MIN_BINS: 5,
    MAX_BINS: 20,

    // Canvas padding
    PADDING: 80, // For histograms
    SCATTER_PADDING: 100, // For scatter plot

    // Visual properties
    LINE_WIDTH: 2,
    POINT_RADIUS: 6,
    BAR_WIDTH_RATIO: 0.8, // Bar width as ratio of bin width

    // Font settings
    FONT_SIZE: 24, // Scaled for high-DPI displays
    FONT_SIZE_TITLE: 28,
    FONT_FAMILY: 'IBM Plex Sans',
  },

  /**
   * UI Interaction
   */
  INTERACTION: {
    // Tooltip positioning
    TOOLTIP_OFFSET: 5, // pixels above target element
    TOOLTIP_EDGE_MARGIN: 5, // pixels from screen edge

    // Token explorer scrolling
    TOKENS_PER_SCROLL: 3, // Number of token cards to scroll at once

    // Scroll button update delay
    SCROLL_BUTTON_UPDATE_DELAY: 300, // milliseconds

    // Position validation tolerance
    POSITION_MATCH_TOLERANCE: 3, // characters - for fuzzy token position matching

    // Search ranges for token position matching
    POSITION_SEARCH_BEFORE: 5, // characters to search before current position
    POSITION_SEARCH_AFTER: 20, // characters to search after current position
  },

  /**
   * Keyboard Shortcuts
   */
  KEYBOARD: {
    UNDO_KEY: 'z',
    REDO_KEY_1: 'y',
    REDO_KEY_2: 'z', // With shift modifier
    PASTE_KEY: 'v',
    CUT_KEY: 'x',
  },

  /**
   * Metric Display Precision
   */
  PRECISION: {
    ENTROPY: 3,
    VARENTROPY: 3,
    SKEWENTROPY: 3,
    PERPLEXITY: 1,
    SURPRISAL: 3,
    PROBABILITY: 3,
    LOG_PROBABILITY: 3,
    DEFAULT: 3,
  },

  /**
   * Fallback Values
   * Used when metrics are invalid or unavailable
   */
  FALLBACKS: {
    ENTROPY: 0.0,
    VARENTROPY: 0.0,
    SKEWENTROPY: 0.0,
    PERPLEXITY: 1.0,
    SURPRISAL: 0.0,
    PROBABILITY: 0.0,
    LOG_PROBABILITY: -10.0,
    METRIC_STRING: '0.000',
    PERPLEXITY_STRING: '1.0',
    LOG_PROB_STRING: '-10.000',
  },

  /**
   * UI State
   */
  STATE: {
    DEFAULT_TAB: 'explorer',
    DEFAULT_SELECTED_INDEX: -1,
  },

  /**
   * Model Display Names
   * Common model name transformations for display
   */
  MODEL_NAMES: {
    'gpt2': 'GPT-2',
    'gpt-2': 'GPT-2',
    'gpt-3': 'GPT-3',
    'gpt-4': 'GPT-4',
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    't5': 'T5',
    'bart': 'BART',
  },
};

// Freeze configuration to prevent accidental modifications
Object.freeze(CONFIG);
Object.freeze(CONFIG.DIMENSIONS);
Object.freeze(CONFIG.ANALYSIS);
Object.freeze(CONFIG.METRICS);
Object.freeze(CONFIG.NORMALIZATION);
Object.freeze(CONFIG.INTENSITY_THRESHOLDS);
Object.freeze(CONFIG.COLORS);
Object.freeze(CONFIG.PERFORMANCE);
Object.freeze(CONFIG.WEBSOCKET);
Object.freeze(CONFIG.TOKENS);
Object.freeze(CONFIG.TOKENIZER);
Object.freeze(CONFIG.TOKENIZER.BPE_ARTIFACTS);
Object.freeze(CONFIG.PLOTS);
Object.freeze(CONFIG.INTERACTION);
Object.freeze(CONFIG.KEYBOARD);
Object.freeze(CONFIG.PRECISION);
Object.freeze(CONFIG.FALLBACKS);
Object.freeze(CONFIG.STATE);
Object.freeze(CONFIG.MODEL_NAMES);
