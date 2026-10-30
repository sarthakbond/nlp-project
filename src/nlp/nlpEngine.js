/**
 * Topic #9: Question Answering System Core NLP Engine
 * 
 * In-Browser Classical NLP Pipeline:
 * 1. Question Processing (Tokenization, Lowercasing, Stop Word Removal, Stemming/Normalization)
 * 2. Vector Space Modeling (Term Frequency - Inverse Document Frequency / TF-IDF)
 * 3. Information Retrieval (Cosine Similarity Ranking across document chunks)
 * 4. Extractive Clause QA (Sentence-level TF-IDF retrieval)
 * 5. Legal Clause Risk Analysis
 * 
 * Serves as both an in-browser zero-latency engine and an instant failover
 * if any cloud/serverless backend is unreachable.
 */

// Common English + Legal stop words
const STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot',
  'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
  'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d',
  'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
  'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s',
  'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
  'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
  'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
  'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
  'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
  'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while',
  'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
  'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves', 'shall', 'herein', 'thereof', 'whereas', 'hereby'
]);

/**
 * Tokenize and normalize text into meaningful terms
 */
export function tokenize(text) {
  if (!text) return [];
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length >= 2 && !STOP_WORDS.has(w));
  return words;
}

/**
 * Generate 1-grams and 2-grams (unigrams + bigrams)
 */
export function extractNGrams(words) {
  const ngrams = [...words];
  for (let i = 0; i < words.length - 1; i++) {
    ngrams.push(`${words[i]} ${words[i + 1]}`);
  }
  return ngrams;
}

/**
 * Topic #9 Text Chunker: Breaks contracts into distinct legal clauses
 */
export function chunkContract(text) {
  if (!text) return [];
  // Split on double newlines or numbered legal clauses (e.g., "1. Term", "Section 4")
  const rawParagraphs = text.split(/\n\s*\n/);
  const chunks = [];

  for (const para of rawParagraphs) {
    const trimmed = para.trim();
    if (trimmed.length > 35) {
      chunks.push(trimmed);
    }
  }

  // Fallback if document has few paragraph breaks
  if (chunks.length < 2) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    let currentChunk = '';
    for (const s of sentences) {
      if ((currentChunk + ' ' + s).length > 400 && currentChunk.length > 50) {
        chunks.push(currentChunk.trim());
        currentChunk = s;
      } else {
        currentChunk += ' ' + s;
      }
    }
    if (currentChunk.trim().length > 30) {
      chunks.push(currentChunk.trim());
    }
  }

  return chunks;
}

/**
 * Build TF-IDF Vector Space Model for document chunks
 */
export function buildTfidfModel(chunks) {
  const vocabulary = new Map(); // term -> index
  const docTermFreqs = []; // list of maps: term -> frequency in chunk
  const docFreq = new Map(); // term -> count of docs containing term
  const N = chunks.length;

  chunks.forEach((chunk, docIdx) => {
    const words = tokenize(chunk);
    const terms = extractNGrams(words);
    const tf = new Map();
    const seenInDoc = new Set();

    for (const term of terms) {
      tf.set(term, (tf.get(term) || 0) + 1);
      if (!seenInDoc.has(term)) {
        seenInDoc.add(term);
        docFreq.set(term, (docFreq.get(term) || 0) + 1);
      }
      if (!vocabulary.has(term)) {
        vocabulary.set(term, vocabulary.size);
      }
    }
    docTermFreqs.push(tf);
  });

  // Calculate Inverse Document Frequency (IDF) with smooth scaling
  // IDF(t) = log((1 + N) / (1 + df(t))) + 1
  const idfWeights = new Map();
  for (const [term, df] of docFreq.entries()) {
    const idf = Math.log((1 + N) / (1 + df)) + 1;
    idfWeights.set(term, idf);
  }

  // Pre-calculate sparse TF-IDF vectors and L2 norms for all chunks
  const chunkVectors = docTermFreqs.map((tf) => {
    const vector = new Map();
    let normSq = 0;

    for (const [term, count] of tf.entries()) {
      // Sublinear TF scaling: 1 + log(tf)
      const tfVal = 1 + Math.log(count);
      const idf = idfWeights.get(term) || 1;
      const weight = tfVal * idf;
      vector.set(term, weight);
      normSq += weight * weight;
    }

    return {
      vector,
      norm: Math.sqrt(normSq) || 1e-9
    };
  });

  // Calculate top terms across the corpus
  const termAvgWeight = [];
  for (const [term, idf] of idfWeights.entries()) {
    const df = docFreq.get(term) || 0;
    termAvgWeight.push({ term, weight: idf * df });
  }
  termAvgWeight.sort((a, b) => b.weight - a.weight);
  const topKeywords = termAvgWeight.slice(0, 12).map(item => item.term);

  return {
    vocabulary,
    idfWeights,
    chunkVectors,
    topKeywords,
    totalTerms: vocabulary.size
  };
}

/**
 * Topic #9 Question Processing & Cosine Similarity Information Retrieval
 */
export function queryTfidf(query, model, chunks, topK = 3) {
  const startTime = performance.now();
  const queryWords = tokenize(query);
  const queryTerms = extractNGrams(queryWords);

  if (queryTerms.length === 0) {
    return {
      answer: 'Please enter a more specific natural language question.',
      retrieved_passages: [],
      stats: { latency_ms: 0, vocab_size: model.totalTerms }
    };
  }

  // 1. Vectorize Query using learned IDF weights
  const queryTf = new Map();
  for (const term of queryTerms) {
    queryTf.set(term, (queryTf.get(term) || 0) + 1);
  }

  const queryVector = new Map();
  let queryNormSq = 0;
  const matchedTokens = [];

  for (const [term, count] of queryTf.entries()) {
    if (model.idfWeights.has(term)) {
      const tfVal = 1 + Math.log(count);
      const idf = model.idfWeights.get(term);
      const weight = tfVal * idf;
      queryVector.set(term, weight);
      queryNormSq += weight * weight;
      matchedTokens.push(term);
    }
  }

  const queryNorm = Math.sqrt(queryNormSq) || 1e-9;

  // 2. Cosine Similarity: dot(q, d) / (||q|| * ||d||)
  const scores = chunks.map((chunk, idx) => {
    const chunkData = model.chunkVectors[idx];
    let dotProduct = 0;
    const overlappingTerms = [];

    for (const [term, qWeight] of queryVector.entries()) {
      if (chunkData.vector.has(term)) {
        const dWeight = chunkData.vector.get(term);
        const termDot = qWeight * dWeight;
        dotProduct += termDot;
        overlappingTerms.push({ term, qWeight, dWeight, contribution: termDot });
      }
    }

    const similarity = dotProduct / (queryNorm * chunkData.norm);
    return {
      chunkIndex: idx,
      score: Math.min(1.0, Math.max(0.0, similarity)),
      dotProduct,
      overlappingTerms,
      chunk
    };
  });

  // 3. Rank descending by Cosine Similarity
  scores.sort((a, b) => b.score - a.score);
  const topResults = scores.slice(0, topK);

  // 4. Extractive Sentence Ranking from top chunk
  const bestChunk = topResults[0]?.chunk || '';
  const extractedAnswer = extractExtractiveSentence(query, bestChunk);

  // 5. Clause Risk Assessment
  const retrievedPassages = topResults.map((r, rank) => ({
    rank: rank + 1,
    chunk_index: r.chunkIndex,
    score: parseFloat(r.score.toFixed(4)),
    content: r.chunk,
    dotProduct: parseFloat(r.dotProduct.toFixed(4)),
    overlappingTerms: r.overlappingTerms,
    risk: assessLegalRisk(r.chunk)
  }));

  const latency = Math.round(performance.now() - startTime);

  return {
    engine: 'Client-Side Web NLP Core Engine',
    query,
    answer: extractedAnswer,
    retrieved_passages: retrievedPassages,
    stats: {
      total_chunks: chunks.length,
      vocab_size: model.totalTerms,
      latency_ms: latency,
      top_terms: model.topKeywords,
      matched_query_tokens: matchedTokens,
      primary_similarity_score: retrievedPassages[0]?.score || 0.0
    }
  };
}

/**
 * Sentence-level Extractive QA using TF-IDF sentence similarity
 */
function extractExtractiveSentence(query, context) {
  if (!context) return 'No matching clause found.';
  const sentences = context
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 20);

  if (sentences.length === 0) return context.slice(0, 300) + '...';

  const queryTerms = new Set(tokenize(query));
  let bestSentence = sentences[0];
  let maxOverlap = -1;

  for (const sentence of sentences) {
    const sTerms = tokenize(sentence);
    let overlap = 0;
    for (const t of sTerms) {
      if (queryTerms.has(t)) overlap++;
    }
    if (overlap > maxOverlap) {
      maxOverlap = overlap;
      bestSentence = sentence;
    }
  }

  return bestSentence;
}

/**
 * Real-World Legal Clause Risk Categorization
 */
export function assessLegalRisk(clauseText) {
  const lower = clauseText.toLowerCase();
  const highRiskTriggers = [
    'uncapped liability', 'indemnify and hold harmless', 'liquidated damages',
    'immediate termination', 'penalties', 'gross negligence', 'irrevocable',
    'perpetual and exclusive', 'non-compete', 'sole discretion'
  ];

  const moderateRiskTriggers = [
    'limitation of liability', 'governing law', 'jurisdiction', 'audit rights',
    'confidentiality', 'warranty disclaimer', 'intellectual property assignment',
    '72 hours', 'indemnification', 'prior written consent'
  ];

  const foundHigh = highRiskTriggers.filter(t => lower.includes(t));
  const foundMod = moderateRiskTriggers.filter(t => lower.includes(t));

  if (foundHigh.length > 0) {
    return {
      level: 'HIGH RISK',
      color: '#B91C1C',
      bgColor: '#FEF2F2',
      borderColor: '#FECACA',
      badge: 'Critical Review Required',
      triggers: foundHigh
    };
  } else if (foundMod.length > 0) {
    return {
      level: 'MODERATE',
      color: '#B45309',
      bgColor: '#FFFBEB',
      borderColor: '#FDE68A',
      badge: 'Attention Recommended',
      triggers: foundMod
    };
  }

  return {
    level: 'LOW RISK',
    color: '#15803D',
    bgColor: '#F0FDF4',
    borderColor: '#BBF7D0',
    badge: 'Standard Contractual Clause',
    triggers: []
  };
}
