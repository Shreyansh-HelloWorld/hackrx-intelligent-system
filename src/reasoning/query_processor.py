# src/reasoning/query_processor.py
# HackRx 6.0 - Query Processing and Reasoning Engine

import asyncio
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from ..utils.config import get_settings
from ..utils.logger import get_logger, log_performance_metric
from ..utils.helpers import clean_text, extract_numbers_from_text, create_prompt_template
from ..core.document_processor import DocumentProcessor, ProcessedDocument
from ..core.embeddings import get_embedding_generator
from ..core.vector_store import get_vector_store, SearchResult
from ..core.llm_handler import get_llm_handler, LLMProvider

logger = get_logger(__name__)


class QueryType(Enum):
    COVERAGE_CHECK = "coverage_check"
    ELIGIBILITY = "eligibility" 
    CONDITIONS = "conditions"
    BENEFITS = "benefits"
    EXCLUSIONS = "exclusions"
    WAITING_PERIOD = "waiting_period"
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    """
    Analysis of a user query
    """
    original_query: str
    query_type: QueryType
    entities: Dict[str, Any]
    intent: str
    keywords: List[str]
    confidence: float
    
    def __post_init__(self):
        if not self.entities:
            self.entities = {}
        if not self.keywords:
            self.keywords = []


@dataclass
class AnswerResult:
    """
    Result of query processing
    """
    question: str
    answer: str
    confidence: float
    reasoning: str
    sources: List[Dict[str, Any]]
    query_analysis: QueryAnalysis
    processing_time: float


class QueryProcessor:
    """
    Main query processing engine that coordinates document analysis and question answering
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.document_processor = DocumentProcessor()
        self.current_document = None
        self.document_cache = {}  # Cache processed documents
        
        # Query classification patterns
        self.query_patterns = {
            QueryType.COVERAGE_CHECK: [
                r"does.*cover", r"is.*covered", r"cover.*surgery", r"cover.*treatment",
                r"covered.*condition", r"policy.*cover"
            ],
            QueryType.ELIGIBILITY: [
                r"eligible", r"qualify", r"age.*limit", r"who.*can", r"requirements"
            ],
            QueryType.CONDITIONS: [
                r"condition", r"requirement", r"terms", r"provided.*that", r"subject.*to"
            ],
            QueryType.BENEFITS: [
                r"benefit", r"limit", r"maximum", r"sum.*insured", r"amount"
            ],
            QueryType.EXCLUSIONS: [
                r"exclusion", r"excluded", r"not.*covered", r"except", r"limitation"
            ],
            QueryType.WAITING_PERIOD: [
                r"waiting.*period", r"wait", r"months", r"days.*after", r"before.*covered"
            ]
        }
    
    async def process_document_queries(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Process multiple queries against a document
        
        Args:
            document_url: URL of document to analyze
            questions: List of questions to answer
        
        Returns:
            List of answers corresponding to questions
        
        Raises:
            RuntimeError: If processing fails
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Processing {len(questions)} queries against document: {document_url}")
            
            # Process document if not cached
            processed_doc = await self._get_or_process_document(document_url)
            
            # Process each question
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                try:
                    answer_result = await self._process_single_query(question, processed_doc)
                    answers.append(answer_result.answer)
                    
                    logger.info(f"✅ Question {i+1} answered (confidence: {answer_result.confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process question {i+1}: {e}")
                    answers.append(f"Sorry, I couldn't process this question: {str(e)}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            log_performance_metric("document_queries", processing_time, len(questions))
            
            logger.info(f"✅ Processed {len(questions)} queries in {processing_time:.2f}s")
            return answers
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Document query processing failed after {processing_time:.2f}s: {e}")
            raise RuntimeError(f"Failed to process document queries: {e}")
    
    async def _get_or_process_document(self, document_url: str) -> ProcessedDocument:
        """
        Get document from cache or process it
        
        Args:
            document_url: URL of document to process
        
        Returns:
            ProcessedDocument instance
        """
        # Check cache first
        if document_url in self.document_cache:
            logger.info("Using cached document")
            return self.document_cache[document_url]
        
        # Process document
        logger.info("Processing new document")
        processed_doc = await self.document_processor.process_document_url(document_url)
        
        # Generate embeddings
        embedding_generator = await get_embedding_generator()
        embeddings = await embedding_generator.generate_document_embeddings(processed_doc)
        
        # Store in vector database
        vector_store = await get_vector_store()
        await vector_store.clear()  # Clear previous document
        await vector_store.add_embeddings(embeddings, processed_doc.document_hash)
        
        # Cache the processed document
        self.document_cache[document_url] = processed_doc
        self.current_document = processed_doc
        
        logger.info(f"✅ Document processed and cached: {len(embeddings)} embeddings created")
        return processed_doc
    
    async def _process_single_query(self, question: str, processed_doc: ProcessedDocument) -> AnswerResult:
        """
        Process a single query against the document
        
        Args:
            question: Question to answer
            processed_doc: Processed document to search
        
        Returns:
            AnswerResult with answer and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze the query
            query_analysis = await self._analyze_query(question)
            logger.debug(f"Query analysis: {query_analysis.query_type.value}, confidence: {query_analysis.confidence}")
            
            # Search for relevant content
            relevant_chunks = await self._search_relevant_content(question, query_analysis)
            
            if not relevant_chunks:
                logger.warning("No relevant content found for query")
                return AnswerResult(
                    question=question,
                    answer="I couldn't find relevant information in the document to answer this question.",
                    confidence=0.0,
                    reasoning="No relevant content found in document search.",
                    sources=[],
                    query_analysis=query_analysis,
                    processing_time=asyncio.get_event_loop().time() - start_time
                )
            
            # Generate answer using LLM
            answer, reasoning, confidence = await self._generate_answer(
                question, query_analysis, relevant_chunks, processed_doc
            )
            
            # Create source references
            sources = [
                {
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "page": chunk.page_number,
                    "similarity_score": chunk.score,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in relevant_chunks[:3]  # Top 3 sources
            ]
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AnswerResult(
                question=question,
                answer=answer,
                confidence=confidence,
                reasoning=reasoning,
                sources=sources,
                query_analysis=query_analysis,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Single query processing failed after {processing_time:.2f}s: {e}")
            raise
    
    async def _analyze_query(self, question: str) -> QueryAnalysis:
        """
        Analyze query to understand intent and extract entities
        
        Args:
            question: Question to analyze
        
        Returns:
            QueryAnalysis with extracted information
        """
        question_lower = question.lower()
        
        # Classify query type
        query_type = QueryType.GENERAL
        max_confidence = 0.0
        
        for qtype, patterns in self.query_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, question_lower))
            confidence = matches / len(patterns)
            
            if confidence > max_confidence:
                max_confidence = confidence
                query_type = qtype
        
        # Extract entities
        entities = {}
        
        # Extract age if mentioned
        numbers = extract_numbers_from_text(question)
        for num in numbers:
            if 18 <= num <= 100:  # Likely age
                entities["age"] = int(num)
                break
        
        # Extract medical terms/procedures
        medical_terms = [
            "surgery", "treatment", "operation", "procedure", "therapy",
            "knee", "hip", "heart", "eye", "cataract", "maternity", "pregnancy"
        ]
        entities["medical_terms"] = [term for term in medical_terms if term in question_lower]
        
        # Extract locations (basic)
        locations = ["pune", "mumbai", "delhi", "bangalore", "hyderabad", "chennai"]
        entities["locations"] = [loc for loc in locations if loc in question_lower]
        
        # Extract time periods
        time_patterns = [
            (r"(\d+)\s*month", "months"),
            (r"(\d+)\s*year", "years"), 
            (r"(\d+)\s*day", "days")
        ]
        
        for pattern, unit in time_patterns:
            match = re.search(pattern, question_lower)
            if match:
                entities["time_period"] = {"value": int(match.group(1)), "unit": unit}
                break
        
        # Extract keywords
        keywords = re.findall(r'\b[a-zA-Z]{3,}\b', question_lower)
        keywords = [word for word in keywords if word not in ['the', 'and', 'for', 'this', 'that', 'what', 'how', 'does']]
        
        return QueryAnalysis(
            original_query=question,
            query_type=query_type,
            entities=entities,
            intent=self._determine_intent(question, query_type),
            keywords=keywords[:10],  # Top 10 keywords
            confidence=max_confidence
        )
    
    def _determine_intent(self, question: str, query_type: QueryType) -> str:
        """
        Determine the specific intent of the query
        
        Args:
            question: Original question
            query_type: Classified query type
        
        Returns:
            Intent description
        """
        question_lower = question.lower()
        
        if "what" in question_lower:
            return "information_request"
        elif "does" in question_lower or "is" in question_lower:
            return "yes_no_question"
        elif "how" in question_lower:
            return "process_question"
        elif "when" in question_lower:
            return "time_question"
        elif "where" in question_lower:
            return "location_question"
        else:
            return "general_inquiry"
    
    async def _search_relevant_content(self, question: str, query_analysis: QueryAnalysis) -> List[SearchResult]:
        """
        Search for relevant content using hybrid search
        
        Args:
            question: Original question
            query_analysis: Analyzed query information
        
        Returns:
            List of relevant search results
        """
        try:
            # Generate query embedding
            embedding_generator = await get_embedding_generator()
            query_embedding = await embedding_generator.generate_query_embedding(question)
            
            # Perform hybrid search
            vector_store = await get_vector_store()
            search_results = await vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=question,
                top_k=self.settings.top_k_results
            )
            
            # Filter results based on query type
            filtered_results = self._filter_results_by_type(search_results, query_analysis)
            
            logger.info(f"Found {len(filtered_results)} relevant chunks for query")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []
    
    def _filter_results_by_type(self, results: List[SearchResult], query_analysis: QueryAnalysis) -> List[SearchResult]:
        """
        Filter search results based on query type
        
        Args:
            results: Search results to filter
            query_analysis: Query analysis information
        
        Returns:
            Filtered search results
        """
        if query_analysis.query_type == QueryType.GENERAL:
            return results
        
        # Define type-specific keywords for filtering
        type_keywords = {
            QueryType.COVERAGE_CHECK: ["cover", "benefit", "include", "treatment", "procedure"],
            QueryType.EXCLUSIONS: ["exclusion", "excluded", "not covered", "except", "limitation"],
            QueryType.WAITING_PERIOD: ["waiting", "period", "months", "days", "after", "before"],
            QueryType.CONDITIONS: ["condition", "requirement", "terms", "provided", "subject"],
            QueryType.BENEFITS: ["benefit", "limit", "maximum", "amount", "sum insured"],
            QueryType.ELIGIBILITY: ["eligible", "qualify", "age", "requirement"]
        }
        
        keywords = type_keywords.get(query_analysis.query_type, [])
        if not keywords:
            return results
        
        # Score results based on keyword presence
        scored_results = []
        for result in results:
            text_lower = result.text.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if keyword_matches > 0:
                # Boost score based on keyword relevance
                boost = keyword_matches / len(keywords)
                result.score = result.score * (1 + boost)
                scored_results.append(result)
        
        # If no keyword matches, return original results
        if not scored_results:
            return results
        
        # Sort by updated scores
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results
    
    async def _generate_answer(self, question: str, query_analysis: QueryAnalysis,
                             relevant_chunks: List[SearchResult], processed_doc: ProcessedDocument) -> Tuple[str, str, float]:
        """
        Generate answer using LLM with relevant context
        
        Args:
            question: Original question
            query_analysis: Query analysis
            relevant_chunks: Relevant document chunks
            processed_doc: Processed document
        
        Returns:
            Tuple of (answer, reasoning, confidence)
        """
        try:
            # Prepare context from relevant chunks
            context_parts = []
            for i, chunk in enumerate(relevant_chunks[:3]):  # Use top 3 chunks
                page_info = f" (Page {chunk.page_number})" if chunk.page_number else ""
                context_parts.append(f"Context {i+1}{page_info}:\n{chunk.text}")
            
            context = "\n\n".join(context_parts)
            
            # Create system prompt for insurance/policy domain
            system_prompt = """You are an expert insurance policy analyst. Your task is to answer questions about insurance policies based on the provided document context.

Instructions:
1. Answer questions accurately based only on the provided context
2. If information is not available in the context, clearly state so
3. Provide specific details like amounts, time periods, and conditions when available
4. For yes/no questions, give a clear answer followed by explanation
5. Reference specific policy clauses or sections when possible
6. Be concise but comprehensive

Important: Only use information from the provided context. Do not make assumptions or provide general insurance knowledge not present in the document."""
            
            # Create user prompt with context and question
            user_prompt = create_prompt_template(
                """Based on the following policy document context, please answer the question:

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based on the policy information above. If the specific information needed to answer the question is not in the context, please state that clearly.""",
                context=context,
                question=question
            )
            
            # Generate response using LLM
            llm_handler = await get_llm_handler()
            
            # Prefer Gemini for better reasoning with insurance documents
            response = await llm_handler.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                preferred_provider=LLMProvider.GEMINI,
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=800
            )
            
            if not response.success:
                raise RuntimeError(f"LLM response failed: {response.error}")
            
            answer = response.content.strip()
            
            # Generate reasoning explanation
            reasoning = self._generate_reasoning(query_analysis, relevant_chunks, response)
            
            # Calculate confidence based on search results and LLM response
            confidence = self._calculate_confidence(relevant_chunks, response, answer)
            
            logger.info(f"Generated answer with confidence {confidence:.2f}")
            return answer, reasoning, confidence
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return (
                "I encountered an error while processing this question. Please try again.",
                f"Error in answer generation: {str(e)}",
                0.0
            )
    
    def _generate_reasoning(self, query_analysis: QueryAnalysis, 
                          relevant_chunks: List[SearchResult], llm_response) -> str:
        """
        Generate explanation of reasoning process
        
        Args:
            query_analysis: Query analysis
            relevant_chunks: Search results used
            llm_response: LLM response object
        
        Returns:
            Reasoning explanation
        """
        reasoning_parts = [
            f"Query Type: {query_analysis.query_type.value}",
            f"Intent: {query_analysis.intent}",
            f"Relevant Sections Found: {len(relevant_chunks)}",
        ]
        
        if relevant_chunks:
            top_sources = [f"Page {chunk.page_number}" for chunk in relevant_chunks[:2] if chunk.page_number]
            if top_sources:
                reasoning_parts.append(f"Primary Sources: {', '.join(top_sources)}")
        
        reasoning_parts.append(f"LLM Provider: {llm_response.provider.value}")
        reasoning_parts.append(f"Processing Time: {llm_response.response_time:.2f}s")
        
        return " | ".join(reasoning_parts)
    
    def _calculate_confidence(self, relevant_chunks: List[SearchResult], 
                           llm_response, answer: str) -> float:
        """
        Calculate confidence score for the answer
        
        Args:
            relevant_chunks: Search results used
            llm_response: LLM response object
            answer: Generated answer
        
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Search result quality
        if relevant_chunks:
            avg_similarity = sum(chunk.score for chunk in relevant_chunks) / len(relevant_chunks)
            confidence_factors.append(min(avg_similarity, 1.0))
        else:
            confidence_factors.append(0.0)
        
        # Factor 2: Number of relevant sources
        source_factor = min(len(relevant_chunks) / 3.0, 1.0)  # Normalize to 3 sources
        confidence_factors.append(source_factor)
        
        # Factor 3: Answer length and detail (not too short, not too long)
        answer_length_factor = 1.0
        if len(answer) < 20:
            answer_length_factor = 0.5  # Very short answers are less confident
        elif len(answer) > 1000:
            answer_length_factor = 0.8  # Very long answers might be less focused
        confidence_factors.append(answer_length_factor)
        
        # Factor 4: LLM response success
        llm_factor = 1.0 if llm_response.success else 0.0
        confidence_factors.append(llm_factor)
        
        # Factor 5: Specific answer indicators
        specific_indicators = ["yes", "no", "covered", "not covered", "excluded", "eligible", "requirement"]
        if any(indicator in answer.lower() for indicator in specific_indicators):
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.8)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.1, 0.2, 0.2]  # Search quality has highest weight
        weighted_confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return min(max(weighted_confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def cleanup(self):
        """
        Cleanup query processor resources
        """
        logger.info("Cleaning up query processor resources")
        
        # Clear document cache
        self.document_cache.clear()
        self.current_document = None
        
        logger.info("✅ Query processor resources cleaned up")