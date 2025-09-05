import streamlit as st
from src.models.grading import grade_answer
from src.models.content import summarize, diagram_prompt, qa_over_content, adaptation_suggestions, generate_concept_diagram
from src.models.recommend import recommend_modules
from src.models.analytics import compute_analytics
import altair as alt
import pandas as pd

# Simple translation dictionary
T = {
    'en': {
        'grading': "AI-Powered Grading System",
        'grading_desc': "Enter a question, reference answer, and your answer. Get instant grading and feedback.",
        'question': "Question",
        'reference': "Reference Answer",
        'your_answer': "Your Answer",
        'grade_btn': "Grade Answer",
        'score': "Score",
        'feedback': "Feedback",
        'content': "Intelligent Course Content Management",
        'content_desc': "Summarize, visualize, and adapt lesson content. Try with your own text!",
        'lesson_text': "Lesson Text",
        'summarize_btn': "Summarize Lesson",
        'summary': "Summary",
        'bullets': "Bullets",
        'diagram_btn': "Generate Diagram Prompt",
        'diagram': "Diagram Prompt",
        'visual_diagram_btn': "Generate Visual Diagram",
        'visual_diagram': "Visual Concept Map",
        'qa': "Bilingual Q&A Assistant",
        'qa_input': "Ask a question about the course content:",
        'qa_btn': "Ask Q&A",
        'answer': "Answer",
        'source': "Source",
        'adapt': "Content Adaptation Suggestions (VARK)",
        'adapt_btn': "Suggest Adaptations",
        'recommend': "Personalized Recommendation System",
        'recommend_desc': "Get learning module recommendations based on region, patient tags, and level.",
        'tags': "Patient Tags (comma-separated)",
        'level': "Level",
        'recommend_btn': "Recommend Modules",
        'next_steps': "Adaptive Next Steps:",
        'analytics': "Analytics Dashboard & Data Visualization",
        'analytics_desc': "View CHW engagement, progress, and regional comparisons for region:",
        'compute_analytics': "Compute Analytics",
        'daily_active': "Daily Active CHWs",
        'regional_scores': "Regional Average Scores",
        'region_stats': "Region Stats:",
        'weekly_active': "Weekly Active CHWs:",
        'avg_session': "Average Session Length (min):",
        'quiz_attempts': "Quiz Attempts:",
        'avg_score': "Average Score:",
        'completion': "Completion Rate:",
        'flagged': "Flagged CHWs (need training):"
    },
    'ki': {
        'grading': "Sisitemu yo Gusuzuma yifashishije AI",
        'grading_desc': "Andika ikibazo, igisubizo cy'icyitegererezo, n'igisubizo cyawe. Sangira amanota n'inama ako kanya.",
        'question': "Ikibazo",
        'reference': "Igisubizo cy'Icyitegererezo",
        'your_answer': "Igisubizo cyawe",
        'grade_btn': "Suzuma Igisubizo",
        'score': "Amanota",
        'feedback': "Inama",
        'content': "Gucunga Ibisobanuro by'Isomo hifashishijwe AI",
        'content_desc': "Sobanura, shushanya, kandi uhindure isomo. Gerageza inyandiko yawe!",
        'lesson_text': "Inyandiko y'Isomo",
        'summarize_btn': "Sobanura Isomo",
        'summary': "Ibisobanuro",
        'bullets': "Ingingo z'ingenzi",
        'diagram_btn': "Tegura Ishusho y'Isano",
        'diagram': "Ishusho y'Isano",
        'visual_diagram_btn': "Tegura Ishusho y'Ubwoba",
        'visual_diagram': "Ishusho y'Ubwoba",
        'qa': "Umufasha wa Q&A w'Indimi ebyiri",
        'qa_input': "Baza ikibazo ku isomo:",
        'qa_btn': "Baza Q&A",
        'answer': "Igisubizo",
        'source': "Aho byakuwe",
        'adapt': "Inama zo Guhindura Isomo (VARK)",
        'adapt_btn': "Saba Inama",
        'recommend': "Sisitemu yo Gutanga Inama z'Isomo",
        'recommend_desc': "Bona inama z'isomo zishingiye ku karere, ibimenyetso by'umurwayi, n'urwego.",
        'tags': "Ibimenyetso by'umurwayi (byandikwe bitandukanyijwe na koma)",
        'level': "Urwego",
        'recommend_btn': "Saba Inama z'Isomo",
        'next_steps': "Intambwe zikurikiraho:",
        'analytics': "Dashboard y'Ibisobanuro n'Ishusho",
        'analytics_desc': "Reba uko abajyanama bakora, intambwe, n'itandukaniro ry'uturere kuri:",
        'compute_analytics': "Bara Ibisobanuro",
        'daily_active': "Abajyanama bakora buri munsi",
        'regional_scores': "Amanota y'Akarere",
        'region_stats': "Ibisobanuro by'Akarere:",
        'weekly_active': "Abajyanama bakora mu cyumweru:",
        'avg_session': "Igihe cy'Inyigisho (min):",
        'quiz_attempts': "Ibizamini Byakozwe:",
        'avg_score': "Amanota (Average):",
        'completion': "Abasoje (%):",
        'flagged': "Abajyanama bakeneye kongererwa ubumenyi:"
    }
}

st.set_page_config(page_title="CHW E-Learning AI Demo", layout="wide")

# Sidebar
st.sidebar.title("Settings")
language = st.sidebar.selectbox("Language", ["English", "Kinyarwanda"], index=0, key="lang")
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West", "Kigali"], index=0, key="region")
lang = "en" if language == "English" else "ki"
if st.sidebar.button("Reset Seed"):
    st.session_state.clear()

# Tabs
tabs = st.tabs([T[lang]['grading'], T[lang]['content'], T[lang]['recommend'], T[lang]['analytics']])

with tabs[0]:
    st.header(T[lang]['grading'])
    st.write(T[lang]['grading_desc'])
    q = st.text_area(T[lang]['question'], key="grade_q")
    ref = st.text_area(T[lang]['reference'], key="grade_ref")
    user = st.text_area(T[lang]['your_answer'], key="grade_user")
    if st.button(T[lang]['grade_btn']):
        try:
            result = grade_answer(q, ref, user, lang)
            # Handle 0-5 scoring
            score = result['score_0_to_5']
            confidence = result['confidence']
            similarity = result['similarity']
            
            # Display score with confidence
            if score == 0:
                st.error(f"{T[lang]['score']}: {score} - Completely off-topic or no understanding")
            elif score <= 2:
                st.warning(f"{T[lang]['score']}: {score}")
            elif score <= 4:
                st.info(f"{T[lang]['score']}: {score}")
            else:
                st.success(f"{T[lang]['score']}: {score}")
            
            # Display confidence and similarity metrics
            col1, col2 = st.columns(2)
            with col1:
                # Confidence display with color coding
                if confidence >= 0.8:
                    st.success(f"**Confidence: {confidence:.1%}** (High)")
                elif confidence >= 0.6:
                    st.info(f"**Confidence: {confidence:.1%}** (Medium)")
                else:
                    st.warning(f"**Confidence: {confidence:.1%}** (Low)")
            
            with col2:
                st.metric("Similarity", f"{similarity:.1%}")
            
            # Display semantic understanding metrics
            if result['similarity'] < 0.3:
                st.warning("⚠️ **Low semantic similarity** - Answer may not address the main concepts")
            elif result['similarity'] < 0.6:
                st.info("ℹ️ **Moderate semantic similarity** - Answer touches on some concepts")
            else:
                st.success("✅ **High semantic similarity** - Answer addresses the main concepts well")
            
            st.write(f"**{T[lang]['feedback']}:**")
            for s in result['suggestions']:
                st.write(f"- {s}")
                
            # Add confidence explanation
            if confidence < 0.5:
                st.info("💡 **Note:** Low confidence means the AI is less certain about this grade. Consider reviewing your answer for clarity.")
        except Exception as e:
            st.error(f"Error grading answer: {e}")

with tabs[1]:
    st.header(T[lang]['content'])
    st.write(T[lang]['content_desc'])
    lesson = st.text_area(T[lang]['lesson_text'], key="lesson_text")
    if st.button(T[lang]['summarize_btn']):
        try:
            summ = summarize(lesson, lang)
            st.write(f"**{T[lang]['summary']}:**", summ['summary'])
            st.write(f"**{T[lang]['bullets']}:**")
            for bullet in summ['bullets']:
                st.write(f"• {bullet}")
        except Exception as e:
            st.error(f"Error summarizing: {e}")
    if st.button(T[lang]['diagram_btn']):
        try:
            diag = diagram_prompt(lesson, lang)
            st.write(f"**{T[lang]['diagram']}:**", diag['prompt'])
        except Exception as e:
            st.error(f"Error generating diagram prompt: {e}")
    
    if st.button(T[lang]['visual_diagram_btn']):
        try:
            with st.spinner("Generating visual diagram..."):
                diagram_image = generate_concept_diagram(lesson, lang)
                if diagram_image:
                    st.write(f"**{T[lang]['visual_diagram']}:**")
                    st.image(f"data:image/png;base64,{diagram_image}", use_column_width=True)
                else:
                    st.error("Failed to generate diagram. Please try again.")
        except Exception as e:
            st.error(f"Error generating visual diagram: {e}")
    st.write("---")
    st.subheader(T[lang]['qa'])
    q = st.text_input(T[lang]['qa_input'], key="qa_q")
    if st.button(T[lang]['qa_btn']):
        try:
            ans = qa_over_content(q, lang)
            st.write(f"**{T[lang]['answer']}:**", ans['answer'])
            if ans.get('source_title'):
                st.caption(f"**{T[lang]['source']}:** {ans['source_title']}")
                if ans.get('focus_area'):
                    st.caption(f"**Focus Area:** {ans['focus_area']}")
                if ans.get('confidence'):
                    st.caption(f"**Confidence:** {ans['confidence']}")
        except Exception as e:
            st.error(f"Error in Q&A: {e}")
    st.write("---")
    st.subheader(T[lang]['adapt'])
    if st.button(T[lang]['adapt_btn']):
        try:
            tips = adaptation_suggestions(lesson, lang)
            for style, tip in tips['suggestions'].items():
                st.write(f"**{style.title()}:** {tip[0]}")
        except Exception as e:
            st.error(f"Error suggesting adaptations: {e}")

with tabs[2]:
    st.header(T[lang]['recommend'])
    st.write(T[lang]['recommend_desc'])
    tags = st.text_input(T[lang]['tags'], key="rec_tags")
    level = st.selectbox(T[lang]['level'], ["basic", "advanced"], index=0, key="rec_level")
    if st.button(T[lang]['recommend_btn']):
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            rec = recommend_modules(region, tag_list, level)
            for m in rec['modules']:
                st.write(f"**{m['module']}** (Score: {m['score']})")
                for r in m['rationales']:
                    st.caption(f"- {r}")
            st.write(f"**{T[lang]['next_steps']}**", rec['next_steps'])
        except Exception as e:
            st.error(f"Error in recommendations: {e}")

with tabs[3]:
    st.header(T[lang]['analytics'])
    st.write(f"{T[lang]['analytics_desc']} {region}.")
    
    # Auto-compute analytics on page load
    try:
        analytics = compute_analytics()
        
        # === KPI DASHBOARD ===
        st.subheader("📊 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total CHWs", 
                analytics['total_chws'],
                delta=f"{analytics['monthly_active']} active this month"
            )
        with col2:
            st.metric(
                "Weekly Active", 
                analytics['weekly_active'],
                delta=f"{analytics['retention_7d']:.1%} retention"
            )
        with col3:
            st.metric(
                "Avg Session Length", 
                f"{analytics['avg_session_length']} min",
                delta=f"Median: {analytics['median_session_length']} min"
            )
        with col4:
            st.metric(
                "Avg Score", 
                f"{analytics['avg_score']:.1f}",
                delta=f"Median: {analytics['median_score']:.1f}"
            )
        
        # === ENGAGEMENT METRICS ===
        st.subheader("📈 Engagement Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Completion Rate", f"{analytics['completion_rate']:.1%}")
        with col2:
            st.metric("High Engagement", f"{analytics['high_engagement_rate']:.1%}")
        with col3:
            st.metric("Total Quiz Attempts", f"{analytics['total_quiz_attempts']:,}")
        
        # === TREND CHARTS ===
        st.subheader("📊 Activity Trends")
        
        # Daily active CHWs trend
        daily_df = pd.DataFrame(analytics['daily_trends'])
        if not daily_df.empty:
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Daily Active CHWs**")
                chart = alt.Chart(daily_df).mark_line(point=True).add_selection(
                    alt.selection_interval()
                ).encode(
                    x='date:T',
                    y='active_chws:Q',
                    tooltip=['date:T', 'active_chws:Q']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            
            with col2:
                st.write("**Average Scores Over Time**")
                chart = alt.Chart(daily_df).mark_line(point=True).encode(
                    x='date:T',
                    y='avg_score:Q',
                    tooltip=['date:T', 'avg_score:Q']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
        
        # === PERFORMANCE DISTRIBUTION ===
        st.subheader("📊 Performance Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Score Distribution**")
            score_data = pd.DataFrame([
                {'Category': 'Excellent (4.5+)', 'Count': analytics['score_distribution']['excellent']},
                {'Category': 'Good (3.5-4.5)', 'Count': analytics['score_distribution']['good']},
                {'Category': 'Average (2.5-3.5)', 'Count': analytics['score_distribution']['average']},
                {'Category': 'Needs Improvement (<2.5)', 'Count': analytics['score_distribution']['needs_improvement']}
            ])
            
            chart = alt.Chart(score_data).mark_bar().encode(
                x='Count:Q',
                y='Category:O',
                color=alt.Color('Category', scale=alt.Scale(range=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])),
                tooltip=['Category', 'Count']
            )
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.write("**Session Length Distribution**")
            session_data = pd.DataFrame([
                {'Category': 'Short (<30min)', 'Count': analytics['session_length_distribution']['short']},
                {'Category': 'Medium (30-60min)', 'Count': analytics['session_length_distribution']['medium']},
                {'Category': 'Long (60+min)', 'Count': analytics['session_length_distribution']['long']}
            ])
            
            chart = alt.Chart(session_data).mark_bar().encode(
                x='Count:Q',
                y='Category:O',
                color=alt.Color('Category', scale=alt.Scale(range=['#ff9ff3', '#54a0ff', '#5f27cd'])),
                tooltip=['Category', 'Count']
            )
            st.altair_chart(chart, use_container_width=True)
        
        # === REGIONAL ANALYSIS ===
        st.subheader("🌍 Regional Performance")
        
        reg_df = pd.DataFrame(analytics['regional_comparison'])
        if not reg_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Average Scores by Region**")
                chart = alt.Chart(reg_df).mark_bar().encode(
                    x='avg_score_mean:Q',
                    y='region:O',
                    color=alt.Color('avg_score_mean', scale=alt.Scale(scheme='viridis')),
                    tooltip=['region', 'avg_score_mean', 'total_chws']
                )
                st.altair_chart(chart, use_container_width=True)
            
            with col2:
                st.write("**Session Length by Region**")
                chart = alt.Chart(reg_df).mark_bar().encode(
                    x='avg_session_length:Q',
                    y='region:O',
                    color=alt.Color('avg_session_length', scale=alt.Scale(scheme='plasma')),
                    tooltip=['region', 'avg_session_length', 'total_chws']
                )
                st.altair_chart(chart, use_container_width=True)
            
            # Regional comparison table
            st.write("**Detailed Regional Comparison**")
            display_df = reg_df[['region', 'total_chws', 'avg_score_mean', 'avg_session_length', 'total_sessions']].round(2)
            display_df.columns = ['Region', 'CHWs', 'Avg Score', 'Avg Session (min)', 'Total Sessions']
            st.dataframe(display_df, use_container_width=True)
        
        # === WEEKLY PATTERNS ===
        st.subheader("📅 Weekly Activity Patterns")
        
        day_data = pd.DataFrame([
            {'Day': day, 'Active CHWs': count} 
            for day, count in analytics['day_of_week_activity'].items()
        ])
        
        chart = alt.Chart(day_data).mark_bar().encode(
            x='Day:O',
            y='Active CHWs:Q',
            color=alt.Color('Active CHWs:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Day', 'Active CHWs']
        )
        st.altair_chart(chart, use_container_width=True)
        
        # === PREDICTIVE ANALYTICS ===
        st.subheader("🔮 Predictive Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**CHWs at Risk**")
            if analytics['at_risk_chws']:
                st.warning(f"⚠️ {len(analytics['at_risk_chws'])} CHWs need attention")
                for chw in analytics['at_risk_chws'][:5]:  # Show first 5
                    st.write(f"• {chw}")
                if len(analytics['at_risk_chws']) > 5:
                    st.write(f"... and {len(analytics['at_risk_chws']) - 5} more")
            else:
                st.success("✅ No CHWs flagged as at-risk")
        
        with col2:
            st.write("**Top Performers**")
            if analytics['top_performers']:
                st.success(f"🏆 {len(analytics['top_performers'])} top performers identified")
                for chw in analytics['top_performers'][:5]:  # Show first 5
                    st.write(f"• {chw}")
                if len(analytics['top_performers']) > 5:
                    st.write(f"... and {len(analytics['top_performers']) - 5} more")
            else:
                st.info("No top performers identified yet")
        
        # === INSIGHTS & RECOMMENDATIONS ===
        st.subheader("💡 Insights & Recommendations")
        
        if analytics['insights']:
            for insight in analytics['insights']:
                st.write(insight)
        else:
            st.info("No specific insights available at this time")
        
        # === CHW PERFORMANCE TABLE ===
        st.subheader("👥 Individual CHW Performance (Last 7 Days)")
        
        if analytics['chw_performance']:
            perf_df = pd.DataFrame(analytics['chw_performance']).T
            perf_df = perf_df.round(2)
            perf_df.columns = ['Avg Session (min)', 'Avg Score', 'Quiz Attempts', 'Sessions']
            perf_df = perf_df.sort_values('Avg Score', ascending=False)
            
            # Add performance indicators
            perf_df['Status'] = perf_df.apply(lambda row: 
                '🟢 Excellent' if row['Avg Score'] >= 4.0 and row['Sessions'] >= 3 else
                '🟡 Good' if row['Avg Score'] >= 3.0 else
                '🔴 Needs Support', axis=1
            )
            
            st.dataframe(perf_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error computing analytics: {e}")
        import traceback
        st.error(traceback.format_exc())
