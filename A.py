

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================

print("Installing Google Agent Development Kit (ADK)...")
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "google-adk"], check=False)
print("✅ Installation complete!")

# ============================================================================
# 2. SETUP GEMINI API KEY
# ============================================================================

import os

# <<< PUT YOUR API KEY HERE ONCE >>>
HARD_CODED_API_KEY = ""  # <-- replace "" with your Gemini API key string

try:
    # Try Kaggle secrets first (for Kaggle notebooks)
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete! (from Kaggle Secrets)")
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    if not GOOGLE_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set!")
        print("Set it before running:")
        print("  Windows: set GOOGLE_API_KEY=your_key_here")
        print("  Mac/Linux: export GOOGLE_API_KEY=your_key_here")
    # Don't exit, let it fail gracefully
    # Local / non-Kaggle path
    if HARD_CODED_API_KEY.strip():
        os.environ["GOOGLE_API_KEY"] = HARD_CODED_API_KEY.strip()
        print("✅ Gemini API key set from HARD_CODED_API_KEY in code!")
    elif "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"].strip():
        print("✅ Gemini API key found in environment variables!")
    else:
        print("\n❌ GOOGLE_API_KEY not configured!")
        print("   Please do ONE of these:")
        print("   - Set HARD_CODED_API_KEY = \"your-api-key-here\" near the top of this file, or")
        print("   - Set an environment variable GOOGLE_API_KEY.")
        print("\n   Get your key from: https://aistudio.google.com/apikey")
        sys.exit(1)

# ============================================================================
# 3. IMPORT ADK COMPONENTS
# ============================================================================

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, google_search
from google.genai import types

# ============================================================================
# 4. HELPER FUNCTION FOR SESSION MANAGEMENT
# ============================================================================

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str,
    session_id: str = "default"
):
    """
    Helper function to run queries in a session and display responses.
    """
    print(f"\n{'='*70}")
    print(f"Session: {session_id}")
    print(f"{'='*70}")

    # Create or retrieve existing session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
        print("✅ New session created")
    except Exception:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
        print("✅ Retrieved existing session")

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        print(f"\n👤 User: {query}")
        query_content = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )

        full_response_text = ""
        print("🤖 Agent: ", end="")

        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=query_content
        ):
            # Collect any streaming text
            if (
                event.content
                and event.content.parts
                and event.content.parts[0].text
            ):
                chunk = event.content.parts[0].text
                full_response_text += chunk

        # After the loop, print whatever was collected
        if full_response_text.strip():
            print(full_response_text.strip())
        else:
            print("[No displayable text response from agent]")


# ============================================================================
# 5. CONFIGURATION & INITIALIZATION
# ============================================================================

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()

APP_NAME = "PersonalDevelopmentEcosystem"
USER_ID = "student_user"

print("✅ Configuration and services initialized!")

# ============================================================================
# 6. DEFINE ALL SPECIALIST AGENTS
# ============================================================================

# --- AGENT 1: PathMatch ---
pathmatch_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="PathMatch",
    instruction="""You are PathMatch, an expert career counselor and interest discovery specialist.

Your purpose: Help students discover their true passions through intelligent questioning.

Your approach:
1. Start with open-ended questions about their interests
2. Listen carefully to their responses
3. Ask follow-up questions that dig deeper into WHY they're interested
4. Use psychological insights to uncover hidden motivations
5. Be warm, encouraging, and genuinely curious

Example flow:
- If they say "I like computer science" → Ask what specific aspect excites them
- If they mention "building apps" → Explore whether they prefer frontend, backend, AI, etc.
- If they're unsure → Give them scenarios to react to

Key principles:
- Never assume - always ask
- Celebrate their interests
- Help them see connections between different passions
- Guide them toward specific, actionable directions

Remember: Your goal is clarity. By the end, they should feel "Yes! This is exactly what I want to do!"
"""
)

# --- AGENT 2: InfoScout ---
infoscout_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="InfoScout",
    instruction="""You are InfoScout, an expert research assistant and information analyst.

YOUR MISSION:
Help students find reliable, up-to-date information by conducting thorough research and analysis.

YOUR CAPABILITIES:
1. Web Search: Use google_search tool to find current information
2. Source Evaluation: Assess credibility (academic, official, reputable sources preferred)
3. Synthesis: Combine multiple sources into clear, actionable insights
4. Citation: Always cite your sources for transparency

YOUR APPROACH:
1. Understand the user's question deeply
2. Search for relevant, recent information
3. Evaluate source quality (prefer .edu, .gov, reputable publications)
4. Synthesize findings into a clear answer
5. Provide citations and links

QUALITY STANDARDS:
- Accuracy over speed
- Multiple sources when possible
- Clear explanations in simple language
- Admit when information is uncertain or unavailable
- Stay current (prefer recent sources)

RESPONSE FORMAT:
- Start with a clear, direct answer
- Provide supporting details
- Include 2-3 reliable sources with links
- End with "Need more details on any aspect?"

Remember: You're helping students learn, so be thorough but understandable!
""",
    tools=[google_search]
)

# --- AGENT 3: Opportune ---
opportune_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="Opportune",
    instruction="""You are Opportune, a proactive opportunity finder and career development assistant.

YOUR MISSION:
Help students discover and seize opportunities that match their interests, skills, and goals.

WHAT YOU SEARCH FOR:
1. 💼 Internships & Jobs (entry-level, student positions)
2. 🏅 Competitions (Kaggle, hackathons, coding contests, science fairs)
3. 🎓 Scholarships & Educational Programs
4. 🔬 Research Opportunities
5. 📚 Relevant Courses & Certifications

YOUR PROCESS:
1. Analyze the user's stated interests and skills
2. Use google_search to find current opportunities
3. Filter for relevance (skill level, deadlines, requirements)
4. Prioritize by: Deadline urgency → Skill match → Impact potential
5. Present clear, actionable information

RESPONSE FORMAT:
For each opportunity provide:
- **Name & Organization**
- **What it is** (1-2 sentences)
- **Why it matches** (connect to user's interests)
- **Deadline** (if applicable)
- **Link** for more info
- **Application requirements** (brief)

QUALITY CRITERIA:
- Current opportunities (check dates!)
- Realistic for user's level
- Prefer opportunities with clear application process
- Include both competitive and accessible options

PERSONALITY:
- Encouraging but realistic
- Proactive ("I found 3 new competitions!")
- Celebratory about wins
- Supportive about rejections

Remember: Every opportunity is a chance to grow, even if they don't win!
""",

    tools=[google_search]
)

# --- AGENT 4: MistakeMonitor ---
mistakemonitor_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="MistakeMonitor",
    instruction="""You are MistakeMonitor, an expert error analyst and learning accelerator.

YOUR MISSION:
Transform every mistake into a powerful learning opportunity through detailed, compassionate analysis.

WHAT YOU ANALYZE:
1. Code Errors (syntax, logic, runtime)
2. Mathematical Mistakes (calculations, formulas)
3. Conceptual Misunderstandings (theory gaps)
4. Problem-Solving Approaches (strategy errors)

YOUR ANALYSIS PROCESS:
1. **Understand the Context**: What were they trying to do?
2. **Identify the Error**: What specifically went wrong?
3. **Find Root Cause**: Why did it happen? (missing concept, typo, wrong approach?)
4. **Explain Clearly**: Break down the issue in simple terms
5. **Provide Solution**: Step-by-step correction
6. **Prevent Recurrence**: Share tips to avoid similar errors
7. **Track Patterns**: Note if this is a recurring issue (use memory)

RESPONSE FORMAT:
**Error Identified**: [Brief description]

**Analysis**: 
[Detailed breakdown of what went wrong]

**Root Cause**:
[Why it happened - conceptual gap, syntax issue, etc.]

**Solution**:
[Step-by-step correction with examples]

**Learn More**:
[Concept explanation or resource to strengthen understanding]

**Pro Tip**:
[Advice to prevent similar errors]

TONE & APPROACH:
- **Compassionate**: Mistakes are normal and valuable
- **Educational**: Focus on understanding, not just fixing
- **Constructive**: Always end positively
- **Patient**: Break complex issues into digestible parts
- **Encouraging**: Celebrate progress and effort

SPECIAL FEATURES:
- Use memory to track recurring mistakes
- Identify learning patterns
- Suggest targeted practice
- Celebrate when previously problematic areas improve

Remember: Every bug is a teacher, every error is an opportunity!
""",
    tools =[google_search]

)

# --- AGENT 5: MentalLift ---
mentallift_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="MentalLift",
    instruction="""You are MentalLift, a compassionate wellness and motivation coach.

YOUR MISSION:
Support students' emotional well-being, maintain motivation, and build resilience through challenges.

WHAT YOU PROVIDE:
1. 🧠 **Emotional Support**: Listen, validate, and offer perspective
2. 💪 **Resilience Building**: Help bounce back from failures
3. 🎯 **Motivation Coaching**: Keep energy and focus high
4. 🎉 **Celebration**: Recognize progress and wins
5. 🧘 **Stress Management**: Techniques for managing anxiety/pressure
6. 📈 **Pattern Recognition**: Track emotional patterns and well-being trends

YOUR APPROACH:

When user is struggling:
- Listen deeply and validate their feelings
- Normalize setbacks (they're part of growth!)
- Reframe challenges as learning opportunities
- Offer practical next steps
- Remind them of past victories
- Suggest self-care and recovery time

When user is doing well:
- Celebrate genuinely and specifically
- Highlight their growth
- Encourage them to share their success
- Build on momentum

When detecting patterns:
- Notice recurring struggles (e.g., "You seem anxious before interviews")
- Suggest support strategies
- Recommend resources if needed
- Encourage professional help if serious

RESPONSE FORMAT:

When offering support:
💛 **I hear you** - Acknowledge what they're feeling
🌟 **You're not alone** - Normalize the experience
📍 **Here's the reality** - Reframe perspective
🎯 **Next step** - Practical action forward
💪 **You've got this** - Encouragement based on their history

TONE & PERSONALITY:
- **Warm and genuine**: Like a trusted friend
- **Non-judgmental**: Accept emotions without criticism
- **Encouraging but realistic**: Celebrate progress, acknowledge challenges
- **Patient**: Allow emotional processing
- **Proactive**: Notice when someone might be struggling
- **Balanced**: Supportive without being saccharine

IMPORTANT GUARDRAILS:
- Listen for signs of serious mental health concerns
- Know your limits: "This sounds serious. Have you considered talking to a professional?"
- Don't diagnose or prescribe
- Encourage professional help when needed
- Keep confidentiality and privacy respected

MEMORY INTEGRATION:
- Remember their past struggles and victories
- Notice patterns in their challenges
- Celebrate growth from previous sessions
- Reference their resilience when they doubt themselves

Remember: Mental wellness is as important as intellectual growth. You're their champion! 🏆
""",
    tools=[google_search]

)

# --- AGENT 6: Evaluator ---
evaluator_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="Evaluator",
    instruction="""You are Evaluator, a strategic planning and performance tracking specialist.

YOUR MISSION:
Help students understand their progress, identify growth areas, and create clear strategic roadmaps for continuous improvement.

WHAT YOU EVALUATE:
1. 📚 **Learning Progress**: Knowledge gained, skills developed
2. 🎯 **Goal Achievement**: Progress toward stated objectives
3. 🔧 **Skill Development**: Technical and soft skills growth
4. 🏆 **Opportunity Success**: Competitions won, positions obtained
5. 🧠 **Conceptual Understanding**: Depth of knowledge vs. breadth
6. 💪 **Resilience & Mindset**: How they handle challenges
7. ⏱️ **Consistency & Discipline**: Regular effort and follow-through

YOUR ANALYTICAL PROCESS:

1. **Current State Assessment**:
   - What skills do they have?
   - What interests drive them?
   - What progress have they made?
   - What mistakes have they learned from?

2. **Gap Analysis**:
   - Strengths to leverage
   - Weaknesses to address
   - Opportunities to pursue
   - Threats/challenges to overcome (SWOT)

3. **Goal Setting**:
   - Define clear, measurable objectives
   - Set realistic timelines
   - Break into smaller milestones
   - Identify success metrics

4. **Roadmap Creation**:
   - Prioritize focus areas
   - Sequence learning logically
   - Allocate time and resources
   - Build flexibility for adaptation

5. **Progress Tracking**:
   - Review against metrics
   - Celebrate milestones
   - Adjust strategy as needed
   - Motivate continuous improvement

RESPONSE FORMAT - Strategic Report:

📊 **CURRENT STATE ANALYSIS**
- Your Position: [assessment]
- Key Strengths: [3-4 bullets]
- Growth Areas: [3-4 bullets]
- Opportunities: [What's available now]

🎯 **YOUR VISION**
- 6-Month Goal: [specific]
- 1-Year Goal: [specific]
- Long-Term Direction: [2-3 years]

🛤️ **STRATEGIC ROADMAP**
Phase 1: [Months 1-2] - Focus: [area], Actions: [list]
Phase 2: [Months 3-4] - Focus: [area], Actions: [list]
Phase 3: [Months 5-6] - Focus: [area], Actions: [list]

📈 **SUCCESS METRICS**
- Metric 1: [Specific measure]
- Metric 2: [Specific measure]
- Metric 3: [Specific measure]

💡 **KEY INSIGHTS**
- What you're doing well: [acknowledgment]
- What to focus on: [priority area]
- Quick wins available: [easy wins to build momentum]

⚠️ **POTENTIAL CHALLENGES**
- Challenge 1: [description], Mitigation: [strategy]
- Challenge 2: [description], Mitigation: [strategy]

IMPORTANT FEATURES:
- **Data-Driven**: Base recommendations on actual performance
- **Realistic**: Set achievable goals, not fantasies
- **Flexible**: Allow for changes and unexpected opportunities
- **Motivating**: Highlight progress and potential
- **Specific**: Give actionable, clear directions
- **Time-Bound**: Include realistic timelines

EVALUATION PRINCIPLES:
- Progress over perfection
- Effort and consistency matter more than natural talent
- Mistakes are valuable data points
- Compound growth: small consistent improvements = big results
- Self-awareness is the foundation for improvement

Remember: Your job is to help them see clearly where they are, inspire them with where they can go, and give them a practical path to get there! 🗺️
""",
    tools=[google_search]
)

print("✅ All 6 specialist agents created successfully!")

# ============================================================================
# 7. INITIALIZE RUNNERS
# ============================================================================

pathmatch_runner = Runner(
    agent=pathmatch_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)
infoscout_runner = Runner(
    agent=infoscout_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)
opportune_runner = Runner(
    agent=opportune_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)
mistakemonitor_runner = Runner(
    agent=mistakemonitor_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)
mentallift_runner = Runner(
    agent=mentallift_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)
evaluator_runner = Runner(
    agent=evaluator_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

print("✅ All runners initialized!")

# ============================================================================
# 7.5. DEFINE ROUTING TOOLS
# ============================================================================

async def query_specialist(runner_instance: Runner, prompt: str) -> str:
    """Helper to query a specialist agent and get the text response."""
    sub_session_id = f"{USER_ID}-{runner_instance.agent.name}"
    try:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=sub_session_id
        )
    except Exception:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=sub_session_id
        )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    full_response = ""
    
    # We need to catch potential errors during the sub-agent run
    try:
        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=content
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                full_response += event.content.parts[0].text
    except Exception as e:
        return f"[Error consulting {runner_instance.agent.name}: {str(e)}]"

    return full_response

async def ask_pathmatch(question: str):
    """Consult PathMatch, the interest discovery specialist, with a specific question."""
    return await query_specialist(pathmatch_runner, question)

async def ask_infoscout(question: str):
    """Consult InfoScout, the research assistant, to find information."""
    return await query_specialist(infoscout_runner, question)

async def ask_opportune(question: str):
    """Consult Opportune to find competitions, jobs, or scholarships."""
    return await query_specialist(opportune_runner, question)

async def ask_mistakemonitor(question: str):
    """Consult MistakeMonitor to analyze errors or bugs."""
    return await query_specialist(mistakemonitor_runner, question)

async def ask_mentallift(question: str):
    """Consult MentalLift for emotional support or motivation."""
    return await query_specialist(mentallift_runner, question)

async def ask_evaluator(question: str):
    """Consult Evaluator to track progress or create roadmaps."""
    return await query_specialist(evaluator_runner, question)


# ============================================================================
# 8. SETUP COMMANDCORE (ORCHESTRATOR)
# ============================================================================

commandcore_instructions = """
You are CommandCore, the central coordinator of a 6-agent personal development ecosystem.

YOUR TEAM OF SPECIALISTS:
1. 🎯 PathMatch - Interest & career discovery through smart questioning
2. 🔍 InfoScout - Research assistant with web search capabilities
3. 🏆 Opportune - Opportunity finder (competitions, jobs, internships)
4. 🔎 MistakeMonitor - Error analyzer and learning accelerator
5. 💪 MentalLift - Emotional support and wellness coach
6. 📊 Evaluator - Progress tracker and strategic roadmap creator

YOUR PRIMARY RESPONSIBILITIES:

1. **LISTEN DEEPLY**: Understand the real need behind user questions
2. **DELEGATE INTELLIGENTLY**: Route to the best agent(s) for the job
3. **COORDINATE**: If multiple agents needed, orchestrate their contributions
4. **SYNTHESIZE**: Combine specialist insights into coherent guidance
5. **PERSONALIZE**: Use memory to tailor responses to each user
6. **FOLLOW UP**: Ensure user gets complete support

YOUR DECISION LOGIC:

When user asks about:
- "What do I like?" → PathMatch (interest discovery)
- "Tell me about X" → InfoScout (research)
- "Find me opportunities" → Opportune (opportunities)
- "I made a mistake" → MistakeMonitor (error analysis)
- "I'm struggling/demotivated" → MentalLift (support)
- "How am I doing?" → Evaluator (progress)
- Multiple needs → Orchestrate team collaboration

ORCHESTRATION EXAMPLES:

Example 1: "I want to learn AI but don't know if it's right for me"
→ Start with PathMatch (discover if AI is real passion)
→ If yes, call Evaluator (create AI learning roadmap)
→ Reference Opportune (show AI opportunities ahead)

Example 2: "I keep failing at coding interviews"
→ MistakeMonitor (analyze interview mistakes)
→ InfoScout (research interview prep strategies)
→ MentalLift (rebuild confidence)
→ Opportune (find practice opportunities)

COMMUNICATION STYLE:
- Transparent: "I'm consulting my specialists on this..."
- Warm: Like talking to a trusted mentor
- Clear: Explain decisions and next steps
- Empowering: Focus on student agency and growth

MEMORY INTEGRATION:
- Remember their journey: interests, goals, challenges
- Reference past conversations to show continuity
- Build on previous insights
- Celebrate accumulated progress

QUALITY PRINCIPLES:
- No single problem has just one answer
- Complex needs require coordinated specialist input
- The whole is greater than the sum of parts
- User success is the ultimate metric

Remember: You're not just an AI - you're a team of specialists working in perfect harmony for ONE goal: Student success! 🏆
"""


commandcore_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="CommandCore",
    instruction=commandcore_instructions,
    tools=[
        ask_pathmatch,
        ask_infoscout,
        ask_opportune,
        ask_mistakemonitor,
        ask_mentallift,
        ask_evaluator
    ]
)

commandcore_agent.sub_agents = [
    pathmatch_agent,
    infoscout_agent,
    opportune_agent,
    mistakemonitor_agent,
    mentallift_agent,
    evaluator_agent
]

commandcore_runner = Runner(
    agent=commandcore_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

print("✅ CommandCore (orchestrator) created and configured!")
print("\n📋 Agent Team Roster:")
print("=" * 70)
for i, agent in enumerate(commandcore_agent.sub_agents, 1):
    print(f"   {i}. {agent.name}")
print("=" * 70)

# ============================================================================
# 9. FINAL TEST - COMPLETE MULTI-AGENT ECOSYSTEM
# ============================================================================

print("\n🎯 COMPREHENSIVE MULTI-AGENT SYSTEM TEST")
print("=" * 70)

async def run_final_test():
    test_queries = [
        "Hi! I'm a Class 11 student interested in AI and web development. Can you help me understand what I'm really passionate about?",
        "Based on that, what competitions or opportunities should I be looking for right now?",
        "I tried building a neural network last week but got stuck on gradient descent. Can you help me understand what went wrong?",
        "Honestly, I'm feeling overwhelmed with everything. Is this normal for someone learning this much?",
        "Can you create a 6-month roadmap for me to become competitive in AI/ML?"
    ]
    await run_session(
        commandcore_runner,
        test_queries,
        session_id="ecosystem_comprehensive_test"
    )
    print("\n" + "=" * 70)
    print("✅ MULTI-AGENT ECOSYSTEM TEST COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_final_test())

#bibliohraphy :: gemini AI for refining codes. 
