

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import json
import operator
from dataclasses import dataclass

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available, falling back to simplified workflow")

# Web search and data tools
try:
    from duckduckgo_search import DDGS
    import requests
    from bs4 import BeautifulSoup
    WEB_TOOLS_AVAILABLE = True
except ImportError:
    WEB_TOOLS_AVAILABLE = False

# Financial data tools
import yfinance as yf
from .memory.memory_pipeline import MemoryPipeline
from .memory.redis_store import LangGraphRedisCheckpointer

logger = logging.getLogger(__name__)



class ResearchState(TypedDict):
    """State definition for the research workflow"""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    session_id: str
    thread_id: str
    user_id: str
    
    # Research pipeline state
    research_stage: str  # 'planning', 'web_search', 'financial_data', 'peer_analysis', 'synthesis', 'complete'
    search_results: List[Dict[str, Any]]
    financial_data: Dict[str, Any]
    peer_data: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    
    # Memory and context
    memory_context: Dict[str, Any]
    sources_found: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    
    # Output
    final_report: str
    citations: List[Dict[str, Any]]
    export_ready: bool


@dataclass
class Source:
    """Structured source information"""
    url: str
    title: str
    content: str
    relevance_score: float
    source_type: str  # 'web', 'financial', 'news', 'report'
    extracted_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'relevance_score': self.relevance_score,
            'source_type': self.source_type,
            'extracted_at': self.extracted_at.isoformat()
        }


class EnhancedResearchAgent:
    """
    Enhanced multi-agent research system using LangGraph for financial analysis
    Features:
    - Multi-step workflow with checkpointing
    - Specialized agents for different research phases
    - Memory integration for context and learning
    - Web search with source deduplication
    - Financial data integration
    - Peer comparison analysis
    - Citation management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced research agent"""
        self.config = config
        self.session_id = None
        self.memory_pipeline = None
        
        # Initialize components based on availability
        if LANGGRAPH_AVAILABLE:
            self._init_langgraph_workflow()
        else:
            self._init_fallback_workflow()
        
        if WEB_TOOLS_AVAILABLE:
            self.ddgs = DDGS()
        
        logger.info("✅ Enhanced Research Agent initialized")
    
    def _init_langgraph_workflow(self):
        """Initialize LangGraph workflow with specialized agents"""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.config.get('model', 'gpt-4-turbo-preview'),
                temperature=0.1,
                api_key=self.config.get('openai_api_key')
            )
            
            # Create the workflow graph
            workflow = StateGraph(ResearchState)
            
            # Add nodes for each research phase
            workflow.add_node("research_planner", self._research_planner_agent)
            workflow.add_node("web_searcher", self._web_search_agent)
            workflow.add_node("financial_analyzer", self._financial_data_agent)
            workflow.add_node("peer_analyzer", self._peer_comparison_agent)
            workflow.add_node("content_synthesizer", self._synthesis_agent)
            workflow.add_node("report_generator", self._report_generation_agent)
            
            # Define the workflow edges
            workflow.set_entry_point("research_planner")
            workflow.add_edge("research_planner", "web_searcher")
            workflow.add_edge("web_searcher", "financial_analyzer")
            workflow.add_edge("financial_analyzer", "peer_analyzer")
            workflow.add_edge("peer_analyzer", "content_synthesizer")
            workflow.add_edge("content_synthesizer", "report_generator")
            workflow.add_edge("report_generator", END)
            
            # Compile with checkpointing
            if hasattr(self, 'redis_checkpointer'):
                self.workflow = workflow.compile(checkpointer=self.redis_checkpointer)
            else:
                self.workflow = workflow.compile(checkpointer=MemorySaver())
            
            self.workflow_type = 'langgraph'
            logger.info("LangGraph workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph workflow: {e}")
            self._init_fallback_workflow()
    
    def _init_fallback_workflow(self):
        """Initialize fallback workflow without LangGraph"""
        self.workflow_type = 'fallback'
        logger.info("Using fallback workflow (no LangGraph)")
    
    def setup_session(self, session_id: str, memory_pipeline: MemoryPipeline):
        """Setup session with memory pipeline"""
        self.session_id = session_id
        self.memory_pipeline = memory_pipeline
        
        # Setup Redis checkpointer if available
        if hasattr(memory_pipeline, 'redis_store'):
            self.redis_checkpointer = LangGraphRedisCheckpointer(memory_pipeline.redis_store)
    
   
    
    async def _research_planner_agent(self, state: ResearchState) -> Dict[str, Any]:
        """Research planning agent - analyzes query and creates research plan"""
        try:
            logger.info(f"🎯 Research Planner: Analyzing query '{state['query']}'")
            
            # Prepare context from memory if available
            if self.memory_pipeline:
                memory_context = await self.memory_pipeline.prepare_agent_context(
                    state['session_id'], 
                    state['thread_id'], 
                    state['query']
                )
                state['memory_context'] = memory_context
            
            # Analyze query for research plan
            planning_prompt = f"""
            Analyze this financial research query and create a structured research plan:
            
            Query: {state['query']}
            
            Memory Context: {state.get('memory_context', {}).get('context_summary', 'No prior context')}
            
            Create a research plan with:
            1. Key search terms for web research
            2. Required financial data points
            3. Peer companies to analyze (if applicable)
            4. Expected analysis depth
            5. Output format requirements
            
            Return as JSON structure.
            """
            
            if LANGGRAPH_AVAILABLE and hasattr(self, 'llm'):
                response = await self.llm.ainvoke([HumanMessage(content=planning_prompt)])
                plan_text = response.content
            else:
                # Fallback planning
                plan_text = self._generate_fallback_plan(state['query'])
            
            # Parse and structure the plan
            research_plan = self._parse_research_plan(plan_text, state['query'])
            
            # Update state
            state['research_stage'] = 'planning_complete'
            state['messages'].append(AIMessage(content=f"Research plan created: {research_plan['summary']}"))
            
            # Store planning checkpoint
            if self.memory_pipeline:
                await self.memory_pipeline.redis_store.save_research_checkpoint(
                    state['session_id'],
                    'research_plan',
                    research_plan
                )
            
            logger.info("✅ Research planning completed")
            return state
            
        except Exception as e:
            logger.error(f"Research planning failed: {e}")
            state['messages'].append(AIMessage(content=f"Research planning encountered an error: {str(e)}"))
            return state
    
    async def _web_search_agent(self, state: ResearchState) -> Dict[str, Any]:
        """Web search agent - performs comprehensive web research"""
        try:
            logger.info("🌐 Web Search Agent: Gathering web sources")
            
            if not WEB_TOOLS_AVAILABLE:
                logger.warning("Web tools not available, skipping web search")
                state['search_results'] = []
                state['research_stage'] = 'web_search_skipped'
                return state
            
            # Get search terms from research plan or generate from query
            search_terms = self._extract_search_terms(state['query'])
            all_sources = []
            
            for term in search_terms[:3]:  # Limit to 3 main search terms
                try:
                    logger.debug(f"Searching for: {term}")
                    
                    # DuckDuckGo search
                    results = self.ddgs.text(
                        keywords=term,
                        region='us-en',
                        safesearch='moderate',
                        max_results=10
                    )
                    
                    for result in results:
                        source = await self._process_web_source(result)
                        if source and source.relevance_score >= 0.5:
                            all_sources.append(source)
                    
                    # Add delay to respect rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    continue
            
            # Deduplicate sources
            unique_sources = self._deduplicate_sources(all_sources)
            
            # Store sources in state and cache
            state['search_results'] = [source.to_dict() for source in unique_sources[:15]]
            state['sources_found'].extend(state['search_results'])
            state['research_stage'] = 'web_search_complete'
            
            # Cache sources for deduplication
            if self.memory_pipeline:
                for source in unique_sources:
                    await self.memory_pipeline.redis_store.cache_source(
                        source.url,
                        source.to_dict()
                    )
            
            logger.info(f"✅ Web search completed: {len(unique_sources)} unique sources found")
            state['messages'].append(AIMessage(content=f"Found {len(unique_sources)} relevant web sources"))
            
            return state
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            state['search_results'] = []
            state['messages'].append(AIMessage(content=f"Web search encountered an error: {str(e)}"))
            return state
    
    async def _process_web_source(self, search_result: Dict[str, Any]) -> Optional[Source]:
        """Process a single web search result"""
        try:
            url = search_result.get('href', '')
            title = search_result.get('title', '')
            snippet = search_result.get('body', '')
            
            if not url or not title:
                return None
            
            # Check if already cached
            if self.memory_pipeline:
                cached = await self.memory_pipeline.redis_store.get_cached_source(url)
                if cached:
                    logger.debug(f"Using cached source: {url}")
                    return Source(
                        url=cached['url'],
                        title=cached['title'],
                        content=cached['content'],
                        relevance_score=cached['relevance_score'],
                        source_type=cached['source_type'],
                        extracted_at=datetime.fromisoformat(cached['extracted_at'])
                    )
            
            # Attempt to fetch full content (with timeout)
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Research Bot)'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                content = soup.get_text()
                content = ' '.join(content.split())  # Clean whitespace
                
                if len(content) < 100:  # Too short, use snippet
                    content = snippet
                    
            except Exception:
                # Fall back to snippet if fetch fails
                content = snippet
            
            # Calculate relevance score
            relevance = self._calculate_relevance(title + ' ' + content, self.current_query)
            
            # Determine source type
            source_type = self._determine_source_type(url, title)
            
            return Source(
                url=url,
                title=title,
                content=content[:2000],  # Limit content length
                relevance_score=relevance,
                source_type=source_type,
                extracted_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.warning(f"Failed to process source {search_result.get('href', 'unknown')}: {e}")
            return None
    
    async def _financial_data_agent(self, state: ResearchState) -> Dict[str, Any]:
        """Financial data agent - fetches and analyzes financial data"""
        try:
            logger.info("📊 Financial Data Agent: Gathering financial metrics")
            
            # Extract stock symbols from query
            symbols = self._extract_stock_symbols(state['query'])
            
            if not symbols:
                logger.info("No stock symbols found, skipping financial data collection")
                state['financial_data'] = {}
                state['research_stage'] = 'financial_data_skipped'
                return state
            
            financial_data = {}
            
            for symbol in symbols[:5]:  # Limit to 5 symbols
                try:
                    logger.debug(f"Fetching data for {symbol}")
                    
                    # Fetch comprehensive data using yfinance
                    ticker = yf.Ticker(symbol)
                    
                    # Get various data points
                    info = ticker.info
                    history = ticker.history(period="1y")
                    financials = ticker.financials
                    balance_sheet = ticker.balance_sheet
                    cash_flow = ticker.cashflow
                    
                    # Structure the data
                    financial_data[symbol] = {
                        'basic_info': {
                            'name': info.get('longName', symbol),
                            'sector': info.get('sector', 'Unknown'),
                            'industry': info.get('industry', 'Unknown'),
                            'market_cap': info.get('marketCap', 0),
                            'current_price': info.get('currentPrice', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'price_to_book': info.get('priceToBook', 0),
                            'dividend_yield': info.get('dividendYield', 0)
                        },
                        'performance': {
                            'ytd_return': self._calculate_ytd_return(history),
                            'volatility': history['Close'].pct_change().std() * (252 ** 0.5),  # Annualized
                            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                            '52_week_low': info.get('fiftyTwoWeekLow', 0),
                            'avg_volume': history['Volume'].mean()
                        },
                        'fundamentals': {
                            'revenue': financials.loc['Total Revenue'].iloc[0] if not financials.empty and 'Total Revenue' in financials.index else 0,
                            'net_income': financials.loc['Net Income'].iloc[0] if not financials.empty and 'Net Income' in financials.index else 0,
                            'total_assets': balance_sheet.loc['Total Assets'].iloc[0] if not balance_sheet.empty and 'Total Assets' in balance_sheet.index else 0,
                            'total_debt': info.get('totalDebt', 0),
                            'free_cash_flow': info.get('freeCashflow', 0)
                        },
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    
                    # Add to sources
                    source = Source(
                        url=f"https://finance.yahoo.com/quote/{symbol}",
                        title=f"{symbol} Financial Data",
                        content=f"Financial data for {info.get('longName', symbol)}",
                        relevance_score=1.0,
                        source_type='financial',
                        extracted_at=datetime.utcnow()
                    )
                    state['sources_found'].append(source.to_dict())
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch financial data for {symbol}: {e}")
                    continue
            
            state['financial_data'] = financial_data
            state['research_stage'] = 'financial_data_complete'
            
            # Store financial data checkpoint
            if self.memory_pipeline:
                await self.memory_pipeline.redis_store.save_research_checkpoint(
                    state['session_id'],
                    'financial_data',
                    financial_data
                )
            
            logger.info(f"✅ Financial data collected for {len(financial_data)} symbols")
            state['messages'].append(AIMessage(content=f"Collected financial data for {len(financial_data)} symbols"))
            
            return state
            
        except Exception as e:
            logger.error(f"Financial data collection failed: {e}")
            state['financial_data'] = {}
            state['messages'].append(AIMessage(content=f"Financial data collection encountered an error: {str(e)}"))
            return state
    
    async def _peer_comparison_agent(self, state: ResearchState) -> Dict[str, Any]:
        """Peer comparison agent - analyzes peer companies"""
        try:
            logger.info("🏢 Peer Analysis Agent: Conducting peer comparison")
            
            financial_data = state.get('financial_data', {})
            if not financial_data:
                logger.info("No financial data available, skipping peer analysis")
                state['peer_data'] = []
                state['research_stage'] = 'peer_analysis_skipped'
                return state
            
            peer_analysis = []
            
            # For each company, find and analyze peers
            for symbol, data in financial_data.items():
                try:
                    sector = data['basic_info']['sector']
                    industry = data['basic_info']['industry']
                    
                    # Get peer companies (simplified - in production, use more sophisticated peer discovery)
                    peers = self._find_peer_companies(symbol, sector, industry)
                    
                    if peers:
                        peer_data = await self._analyze_peer_metrics(symbol, data, peers)
                        peer_analysis.append(peer_data)
                    
                except Exception as e:
                    logger.warning(f"Peer analysis failed for {symbol}: {e}")
                    continue
            
            state['peer_data'] = peer_analysis
            state['research_stage'] = 'peer_analysis_complete'
            
            # Store peer analysis checkpoint
            if self.memory_pipeline:
                await self.memory_pipeline.redis_store.save_research_checkpoint(
                    state['session_id'],
                    'peer_analysis',
                    peer_analysis
                )
            
            logger.info(f"✅ Peer analysis completed for {len(peer_analysis)} companies")
            state['messages'].append(AIMessage(content=f"Completed peer analysis for {len(peer_analysis)} companies"))
            
            return state
            
        except Exception as e:
            logger.error(f"Peer analysis failed: {e}")
            state['peer_data'] = []
            state['messages'].append(AIMessage(content=f"Peer analysis encountered an error: {str(e)}"))
            return state
    
    async def _synthesis_agent(self, state: ResearchState) -> Dict[str, Any]:
        """Content synthesis agent - combines all research findings"""
        try:
            logger.info("🔬 Synthesis Agent: Combining research findings")
            
            # Gather all available data
            web_sources = state.get('search_results', [])
            financial_data = state.get('financial_data', {})
            peer_data = state.get('peer_data', [])
            memory_context = state.get('memory_context', {})
            
            # Perform synthesis analysis
            analysis = {
                'data_completeness': {
                    'web_sources_count': len(web_sources),
                    'financial_symbols_count': len(financial_data),
                    'peer_analyses_count': len(peer_data),
                    'memory_insights_available': bool(memory_context.get('relevant_memories'))
                },
                'key_findings': [],
                'confidence_assessment': {},
                'synthesis_insights': [],
                'data_quality_score': 0.0
            }
            
            # Analyze web sources
            if web_sources:
                analysis['key_findings'].append(f"Found {len(web_sources)} relevant web sources")
                high_relevance_sources = [s for s in web_sources if s.get('relevance_score', 0) >= 0.8]
                if high_relevance_sources:
                    analysis['synthesis_insights'].append(f"High-quality sources available ({len(high_relevance_sources)} sources with >80% relevance)")
            
            # Analyze financial data
            if financial_data:
                for symbol, data in financial_data.items():
                    company_name = data['basic_info']['name']
                    pe_ratio = data['basic_info']['pe_ratio']
                    market_cap = data['basic_info']['market_cap']
                    
                    analysis['key_findings'].append(f"{company_name} ({symbol}): P/E {pe_ratio:.2f}, Market Cap ${market_cap:,.0f}")
                    
                    # Assess performance
                    ytd_return = data['performance']['ytd_return']
                    if ytd_return > 0.1:
                        analysis['synthesis_insights'].append(f"{symbol} showing strong YTD performance (+{ytd_return:.1%})")
                    elif ytd_return < -0.1:
                        analysis['synthesis_insights'].append(f"{symbol} showing weak YTD performance ({ytd_return:.1%})")
            
            # Analyze peer comparisons
            if peer_data:
                for peer_analysis in peer_data:
                    symbol = peer_analysis.get('primary_symbol', 'Unknown')
                    ranking = peer_analysis.get('peer_ranking', {})
                    if ranking:
                        total_rank = ranking.get('overall_rank', 'N/A')
                        analysis['synthesis_insights'].append(f"{symbol} ranks {total_rank} among sector peers")
            
            # Calculate data quality score
            quality_factors = []
            if web_sources:
                quality_factors.append(min(1.0, len(web_sources) / 10))  # Up to 10 sources = 1.0
            if financial_data:
                quality_factors.append(min(1.0, len(financial_data) / 3))  # Up to 3 symbols = 1.0
            if peer_data:
                quality_factors.append(0.3)  # Peer data adds 0.3
            
            analysis['data_quality_score'] = sum(quality_factors) / max(1, len(quality_factors))
            
            # Confidence assessment
            analysis['confidence_assessment'] = {
                'data_completeness': 'high' if analysis['data_quality_score'] >= 0.8 else 'medium' if analysis['data_quality_score'] >= 0.5 else 'low',
                'source_reliability': 'high' if len([s for s in web_sources if s.get('relevance_score', 0) >= 0.7]) >= 3 else 'medium',
                'analysis_depth': 'comprehensive' if financial_data and peer_data else 'basic'
            }
            
            state['analysis_results'] = analysis
            state['research_stage'] = 'synthesis_complete'
            
            # Store synthesis results
            if self.memory_pipeline:
                await self.memory_pipeline.redis_store.save_research_checkpoint(
                    state['session_id'],
                    'synthesis_results',
                    analysis
                )
                
                # Store key insights in long-term memory
                for insight in analysis['synthesis_insights']:
                    await self.memory_pipeline.store_research_insight(
                        state['session_id'],
                        state['user_id'],
                        insight,
                        web_sources + [{'url': 'financial_data', 'title': 'Financial Analysis'}],
                        'synthesis'
                    )
            
            logger.info("✅ Synthesis completed")
            state['messages'].append(AIMessage(content=f"Synthesis completed with {analysis['data_quality_score']:.1%} data quality"))
            
            return state
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            state['analysis_results'] = {'error': str(e)}
            state['messages'].append(AIMessage(content=f"Synthesis encountered an error: {str(e)}"))
            return state
    
    async def _report_generation_agent(self, state: ResearchState) -> Dict[str, Any]:
        """Report generation agent - creates final comprehensive report"""
        try:
            logger.info("📝 Report Generator: Creating comprehensive report")
            
            # Gather all components for the report
            query = state['query']
            web_sources = state.get('search_results', [])
            financial_data = state.get('financial_data', {})
            peer_data = state.get('peer_data', [])
            analysis = state.get('analysis_results', {})
            memory_context = state.get('memory_context', {})
            
            # Generate comprehensive report
            report_sections = []
            
            # 1. Executive Summary
            report_sections.append("# Financial Research Report")
            report_sections.append(f"**Query:** {query}")
            report_sections.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report_sections.append(f"**Data Quality Score:** {analysis.get('data_quality_score', 0):.1%}")
            report_sections.append("")
            
            # 2. Key Findings
            report_sections.append("## Executive Summary")
            key_findings = analysis.get('key_findings', [])
            if key_findings:
                for finding in key_findings[:5]:  # Top 5 findings
                    report_sections.append(f"- {finding}")
            else:
                report_sections.append("- No specific findings available")
            report_sections.append("")
            
            # 3. Financial Analysis
            if financial_data:
                report_sections.append("## Financial Analysis")
                for symbol, data in financial_data.items():
                    company_name = data['basic_info']['name']
                    report_sections.append(f"### {company_name} ({symbol})")
                    
                    # Basic metrics
                    report_sections.append("**Key Metrics:**")
                    report_sections.append(f"- Market Cap: ${data['basic_info']['market_cap']:,.0f}")
                    report_sections.append(f"- P/E Ratio: {data['basic_info']['pe_ratio']:.2f}")
                    report_sections.append(f"- Price-to-Book: {data['basic_info']['price_to_book']:.2f}")
                    
                    # Performance
                    ytd_return = data['performance']['ytd_return']
                    volatility = data['performance']['volatility']
                    report_sections.append(f"- YTD Return: {ytd_return:.1%}")
                    report_sections.append(f"- Volatility (Annualized): {volatility:.1%}")
                    
                    # Fundamentals
                    revenue = data['fundamentals']['revenue']
                    net_income = data['fundamentals']['net_income']
                    if revenue and net_income:
                        net_margin = (net_income / revenue) * 100
                        report_sections.append(f"- Net Profit Margin: {net_margin:.2f}%")
                    
                    report_sections.append("")
            
            # 4. Peer Comparison
            if peer_data:
                report_sections.append("## Peer Comparison Analysis")
                for peer_analysis in peer_data:
                    symbol = peer_analysis.get('primary_symbol', 'Unknown')
                    ranking = peer_analysis.get('peer_ranking', {})
                    
                    report_sections.append(f"### {symbol} Peer Analysis")
                    if ranking:
                        report_sections.append(f"- Overall Peer Ranking: {ranking.get('overall_rank', 'N/A')}")
                        report_sections.append(f"- Valuation Ranking: {ranking.get('valuation_rank', 'N/A')}")
                        report_sections.append(f"- Performance Ranking: {ranking.get('performance_rank', 'N/A')}")
                    report_sections.append("")
            
            # 5. Market Insights
            synthesis_insights = analysis.get('synthesis_insights', [])
            if synthesis_insights:
                report_sections.append("## Market Insights")
                for insight in synthesis_insights:
                    report_sections.append(f"- {insight}")
                report_sections.append("")
            
            # 6. Data Sources and Methodology
            report_sections.append("## Data Sources and Methodology")
            report_sections.append("**Sources Used:**")
            
            # Financial data sources
            if financial_data:
                report_sections.append(f"- Financial Data: Yahoo Finance API ({len(financial_data)} companies)")
            
            # Web sources
            if web_sources:
                report_sections.append(f"- Web Research: {len(web_sources)} sources")
                high_quality_sources = [s for s in web_sources if s.get('relevance_score', 0) >= 0.8]
                if high_quality_sources:
                    report_sections.append(f"  - High relevance sources: {len(high_quality_sources)}")
            
            # Memory context
            if memory_context.get('relevant_memories'):
                memory_count = len(memory_context['relevant_memories'])
                report_sections.append(f"- Historical Analysis Memory: {memory_count} relevant insights")
            
            report_sections.append("")
            
            # 7. Confidence Assessment
            confidence = analysis.get('confidence_assessment', {})
            if confidence:
                report_sections.append("## Confidence Assessment")
                report_sections.append(f"- Data Completeness: {confidence.get('data_completeness', 'unknown').title()}")
                report_sections.append(f"- Source Reliability: {confidence.get('source_reliability', 'unknown').title()}")
                report_sections.append(f"- Analysis Depth: {confidence.get('analysis_depth', 'unknown').title()}")
                report_sections.append("")
            
            # 8. Limitations and Disclaimers
            report_sections.append("## Limitations and Disclaimers")
            report_sections.append("- This analysis is based on publicly available data and should not be considered investment advice")
            report_sections.append("- Market conditions and company fundamentals can change rapidly")
            report_sections.append("- Past performance does not guarantee future results")
            report_sections.append("- Additional due diligence is recommended before making investment decisions")
            
            # Combine all sections
            final_report = "\n".join(report_sections)
            
            # Prepare citations
            citations = []
            for i, source in enumerate(web_sources[:10], 1):  # Top 10 sources
                citations.append({
                    'id': i,
                    'url': source['url'],
                    'title': source['title'],
                    'relevance_score': source.get('relevance_score', 0),
                    'source_type': source.get('source_type', 'web'),
                    'accessed_at': source.get('extracted_at', datetime.utcnow().isoformat())
                })
            
            # Update state
            state['final_report'] = final_report
            state['citations'] = citations
            state['export_ready'] = True
            state['research_stage'] = 'complete'
            
            # Store final report
            if self.memory_pipeline:
                await self.memory_pipeline.redis_store.save_research_checkpoint(
                    state['session_id'],
                    'final_report',
                    {
                        'report': final_report,
                        'citations': citations,
                        'word_count': len(final_report.split()),
                        'completed_at': datetime.utcnow().isoformat()
                    }
                )
                
                # Store the complete analysis as a memory
                await self.memory_pipeline.store_research_insight(
                    state['session_id'],
                    state['user_id'],
                    f"Comprehensive analysis completed for: {query}",
                    citations,
                    'final_report'
                )
            
            logger.info("✅ Final report generated successfully")
            word_count = len(final_report.split())
            state['messages'].append(AIMessage(content=f"Comprehensive report generated ({word_count:,} words)"))
            
            return state
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state['final_report'] = f"Report generation failed: {str(e)}"
            state['citations'] = []
            state['export_ready'] = False
            state['messages'].append(AIMessage(content=f"Report generation encountered an error: {str(e)}"))
            return state
    
    # ==========================================
    # MAIN WORKFLOW EXECUTION
    # ==========================================
    
    async def conduct_research(self, query: str, session_id: str, user_id: str = None) -> Dict[str, Any]:
        """Main entry point for conducting comprehensive research"""
        try:
            self.current_query = query  # Store for relevance calculations
            
            # Initialize state
            initial_state = ResearchState(
                messages=[HumanMessage(content=query)],
                query=query,
                session_id=session_id,
                thread_id=f"thread_{session_id}_{datetime.utcnow().timestamp()}",
                user_id=user_id or session_id,
                research_stage='initialized',
                search_results=[],
                financial_data={},
                peer_data=[],
                analysis_results={},
                memory_context={},
                sources_found=[],
                confidence_scores={},
                final_report='',
                citations=[],
                export_ready=False
            )
            
            if self.workflow_type == 'langgraph' and hasattr(self, 'workflow'):
                # Use LangGraph workflow
                logger.info("🚀 Starting LangGraph research workflow")
                
                config = {
                    "configurable": {
                        "thread_id": initial_state['thread_id']
                    }
                }
                
                # Execute the workflow
                final_state = await self.workflow.ainvoke(initial_state, config=config)
                
                return {
                    'success': True,
                    'report': final_state.get('final_report', ''),
                    'citations': final_state.get('citations', []),
                    'research_stage': final_state.get('research_stage', 'unknown'),
                    'sources_count': len(final_state.get('sources_found', [])),
                    'data_quality_score': final_state.get('analysis_results', {}).get('data_quality_score', 0),
                    'session_id': session_id,
                    'export_ready': final_state.get('export_ready', False)
                }
            
            else:
                # Use fallback workflow
                logger.info("🔄 Starting fallback research workflow")
                
                # Execute agents sequentially
                state = initial_state
                state = await self._research_planner_agent(state)
                state = await self._web_search_agent(state)
                state = await self._financial_data_agent(state)
                state = await self._peer_comparison_agent(state)
                state = await self._synthesis_agent(state)
                state = await self._report_generation_agent(state)
                
                return {
                    'success': True,
                    'report': state.get('final_report', ''),
                    'citations': state.get('citations', []),
                    'research_stage': state.get('research_stage', 'complete'),
                    'sources_count': len(state.get('sources_found', [])),
                    'data_quality_score': state.get('analysis_results', {}).get('data_quality_score', 0),
                    'session_id': session_id,
                    'export_ready': state.get('export_ready', False)
                }
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'report': f"Research failed: {str(e)}",
                'citations': [],
                'research_stage': 'failed',
                'sources_count': 0,
                'session_id': session_id,
                'export_ready': False
            }
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _generate_fallback_plan(self, query: str) -> str:
        """Generate a basic research plan without LLM"""
        return json.dumps({
            'search_terms': self._extract_search_terms(query),
            'analysis_type': 'comprehensive',
            'expected_sources': 15,
            'financial_data_required': True,
            'peer_analysis_required': True
        })
    
    def _parse_research_plan(self, plan_text: str, query: str) -> Dict[str, Any]:
        """Parse research plan from text"""
        try:
            if plan_text.strip().startswith('{'):
                return json.loads(plan_text)
            else:
                # Fallback parsing
                return {
                    'search_terms': self._extract_search_terms(query),
                    'summary': 'Basic research plan created',
                    'analysis_depth': 'comprehensive'
                }
        except:
            return {
                'search_terms': self._extract_search_terms(query),
                'summary': 'Basic research plan created',
                'analysis_depth': 'comprehensive'
            }
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Basic keyword extraction
        base_terms = [query]
        
        # Add financial context
        if any(word in query.lower() for word in ['stock', 'company', 'financial']):
            base_terms.append(f"{query} financial analysis")
            base_terms.append(f"{query} stock performance")
        
        # Add sector analysis if company mentioned
        symbols = self._extract_stock_symbols(query)
        if symbols:
            for symbol in symbols:
                base_terms.append(f"{symbol} sector analysis")
        
        return base_terms[:5]  # Limit to 5 terms
    
    def _extract_stock_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        import re
        
        # Look for stock symbol patterns (2-5 uppercase letters)
        pattern = r'\b[A-Z]{2,5}\b'
        potential_symbols = re.findall(pattern, query)
        
        # Filter out common words that might match the pattern
        excluded_words = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'INTO', 'NYSE', 'NASDAQ'}
        symbols = [s for s in potential_symbols if s not in excluded_words]
        
        return symbols[:5]  # Limit to 5 symbols
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Basic keyword matching
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        
        base_score = matches / max(len(query_words), 1)
        
        # Boost for financial terms
        financial_terms = ['financial', 'earnings', 'revenue', 'profit', 'stock', 'market', 'analysis']
        financial_matches = sum(1 for term in financial_terms if term in content_lower)
        financial_boost = financial_matches * 0.1
        
        return min(1.0, base_score + financial_boost)
    
    def _determine_source_type(self, url: str, title: str) -> str:
        """Determine the type of source"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        if any(domain in url_lower for domain in ['finance.yahoo.com', 'bloomberg.com', 'marketwatch.com']):
            return 'financial'
        elif any(domain in url_lower for domain in ['news', 'reuters', 'cnbc', 'wsj']):
            return 'news'
        elif any(term in title_lower for term in ['report', 'analysis', 'research']):
            return 'report'
        else:
            return 'web'
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources"""
        seen_urls = set()
        unique_sources = []
        
        # Sort by relevance first
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        
        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        return unique_sources
    
    def _calculate_ytd_return(self, history_df) -> float:
        """Calculate year-to-date return"""
        try:
            if history_df.empty:
                return 0.0
            
            # Get first price of current year
            current_year = datetime.now().year
            year_start = f"{current_year}-01-01"
            
            # Find first available price of the year
            year_data = history_df[history_df.index >= year_start]
            if year_data.empty:
                # Fall back to full period return
                first_price = history_df['Close'].iloc[0]
            else:
                first_price = year_data['Close'].iloc[0]
            
            latest_price = history_df['Close'].iloc[-1]
            
            return (latest_price - first_price) / first_price
        except:
            return 0.0
    
    def _find_peer_companies(self, symbol: str, sector: str, industry: str) -> List[str]:
        """Find peer companies (simplified implementation)"""
        # This is a simplified implementation
        # In production, you'd use a more sophisticated peer discovery service
        
        # Common peers by sector (hardcoded for demonstration)
        sector_peers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
        }
        
        peers = sector_peers.get(sector, [])
        
        # Remove the target symbol from peers
        if symbol in peers:
            peers.remove(symbol)
        
        return peers[:4]  # Return top 4 peers
    
    async def _analyze_peer_metrics(self, primary_symbol: str, primary_data: Dict[str, Any], peer_symbols: List[str]) -> Dict[str, Any]:
        """Analyze metrics against peer companies"""
        try:
            peer_analysis = {
                'primary_symbol': primary_symbol,
                'peer_symbols': peer_symbols,
                'peer_metrics': {},
                'peer_ranking': {},
                'comparison_insights': []
            }
            
            # Get peer data
            peer_metrics = []
            for peer_symbol in peer_symbols:
                try:
                    ticker = yf.Ticker(peer_symbol)
                    info = ticker.info
                    
                    metrics = {
                        'symbol': peer_symbol,
                        'pe_ratio': info.get('trailingPE', 0),
                        'market_cap': info.get('marketCap', 0),
                        'price_to_book': info.get('priceToBook', 0),
                        'roe': info.get('returnOnEquity', 0)
                    }
                    peer_metrics.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Failed to get peer data for {peer_symbol}: {e}")
                    continue
            
            if not peer_metrics:
                return peer_analysis
            
            # Add primary company metrics
            primary_metrics = {
                'symbol': primary_symbol,
                'pe_ratio': primary_data['basic_info']['pe_ratio'],
                'market_cap': primary_data['basic_info']['market_cap'],
                'price_to_book': primary_data['basic_info']['price_to_book'],
                'roe': 0  # Would need to calculate from financials
            }
            
            all_metrics = [primary_metrics] + peer_metrics
            peer_analysis['peer_metrics'] = all_metrics
            
            # Ranking analysis
            if len(all_metrics) >= 2:
                # Rank by P/E ratio (lower is better)
                pe_sorted = sorted([m for m in all_metrics if m['pe_ratio'] > 0], key=lambda x: x['pe_ratio'])
                primary_pe_rank = next((i + 1 for i, m in enumerate(pe_sorted) if m['symbol'] == primary_symbol), 0)
                
                # Rank by market cap (higher is better)
                mcap_sorted = sorted(all_metrics, key=lambda x: x['market_cap'], reverse=True)
                primary_mcap_rank = next((i + 1 for i, m in enumerate(mcap_sorted) if m['symbol'] == primary_symbol), 0)
                
                peer_analysis['peer_ranking'] = {
                    'pe_rank': f"{primary_pe_rank}/{len(pe_sorted)}" if pe_sorted else "N/A",
                    'market_cap_rank': f"{primary_mcap_rank}/{len(all_metrics)}",
                    'overall_rank': f"{(primary_pe_rank + primary_mcap_rank) // 2}/{len(all_metrics)}" if pe_sorted else f"{primary_mcap_rank}/{len(all_metrics)}"
                }
                
                # Generate insights
                if primary_pe_rank <= len(pe_sorted) // 2:
                    peer_analysis['comparison_insights'].append(f"{primary_symbol} has attractive valuation vs peers")
                if primary_mcap_rank <= len(all_metrics) // 2:
                    peer_analysis['comparison_insights'].append(f"{primary_symbol} is among the larger companies in peer group")
            
            return peer_analysis
            
        except Exception as e:
            logger.error(f"Peer analysis failed: {e}")
            return {
                'primary_symbol': primary_symbol,
                'error': str(e)
            }


# Export main class
__all__ = ['EnhancedResearchAgent', 'ResearchState', 'Source']