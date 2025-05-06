# Doc Dock - Company-specific document template manager with AI assistance

import streamlit as st
from datetime import datetime
import base64

def doc_dock_ui(client=None, token_manager=None):
    """
    Implements the Doc Dock UI for uploading company templates and providing AI assistance.
    
    Args:
        client: OpenAI client for AI interactions
        token_manager: TokenManager for tracking API usage
    """
    
    st.subheader("📄 Doc Dock")
    st.markdown("""
    Upload company-specific templates and get AI assistance with document generation, 
    formatting, and content suggestions.
    """)
    
    # Initialize session state for document templates if it doesn't exist
    if 'doc_templates' not in st.session_state:
        st.session_state['doc_templates'] = []
        
    if 'doc_chat_history' not in st.session_state:
        st.session_state['doc_chat_history'] = []
        
    # Create two columns - one for template management, one for chat
    doc_col1, doc_col2 = st.columns([1, 1])
    
    with doc_col1:
        st.markdown("### Upload Templates")
        
        # Template uploader
        uploaded_template = st.file_uploader(
            "Upload company templates (Excel, Word, PowerPoint, PDF)", 
            type=['xlsx', 'xls', 'docx', 'pptx', 'pdf', 'txt'],
            key="template_uploader"
        )
        
        if uploaded_template is not None:
            # Extract file info
            file_details = {
                "name": uploaded_template.name,
                "type": uploaded_template.type,
                "size": uploaded_template.size,
                "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": uploaded_template.getvalue()
            }
            
            # Save to session state if not already there
            existing_names = [t["name"] for t in st.session_state['doc_templates']]
            if uploaded_template.name not in existing_names:
                st.session_state['doc_templates'].append(file_details)
                st.success(f"Template '{uploaded_template.name}' uploaded successfully!")
            else:
                st.info(f"Template '{uploaded_template.name}' already exists. Please rename your file to upload a new version.")
        
        # Display saved templates
        if st.session_state['doc_templates']:
            st.markdown("### Your Templates")
            for i, template in enumerate(st.session_state['doc_templates']):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i+1}. {template['name']} ({template['type']})")                        
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state['doc_templates'].pop(i)
                        st.rerun()
                        
                # Show download button for each template
                if st.button("Download", key=f"download_{i}"):
                    content = template['content']
                    b64 = base64.b64encode(content).decode()
                    href = f'<a href="data:{template["type"]};base64,{b64}" download="{template["name"]}">Click here to download</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No templates uploaded yet. Upload a template to get started.")
    
    with doc_col2:
        st.markdown("### AI Template Assistant")
        st.markdown("Chat with AI about your templates and get assistance.")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state['doc_chat_history']:
                if message['role'] == 'user':
                    st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
                else:  # assistant
                    st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>AI Assistant:</b><br>{message['content']}</div>", unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_area("Type your question or request here:", key="doc_chat_input", height=100)
        
        # Select template context for the chat
        if st.session_state['doc_templates']:
            template_names = [t['name'] for t in st.session_state['doc_templates']]
            selected_template = st.selectbox("Select template for context:", ["None"] + template_names)
        else:
            selected_template = "None"
        
        # Submit button
        if st.button("Submit", key="doc_chat_submit") and user_input:
            # Add user message to history
            st.session_state['doc_chat_history'].append({"role": "user", "content": user_input})
            
            # Prepare context for AI
            prompt = f"The user is asking about document templates: {user_input}\n\n"
            
            # Add template context if selected
            if selected_template != "None":
                template_index = [t['name'] for t in st.session_state['doc_templates']].index(selected_template)
                template_info = st.session_state['doc_templates'][template_index]
                prompt += f"Context: The user is working with a template named '{template_info['name']}' of type {template_info['type']}.\n"
            
            # Generate AI response
            try:
                if client:
                    # Ask the AI about the document
                    api_response = client.chat.completions.create(
                        model="gpt-4o",  # Latest model
                        messages=[
                            {"role": "system", "content": "You are a document template assistant. Help the user with formatting, content suggestions, best practices, and other document-related questions. If they're asking about a specific template, provide tailored advice for that template type."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500
                    )
                    
                    # Extract response
                    ai_response = api_response.choices[0].message.content
                    
                    # Track token usage if token manager is provided
                    if token_manager:
                        token_manager.track_usage(api_response)
                    
                    # Add to history
                    st.session_state['doc_chat_history'].append({"role": "assistant", "content": ai_response})
                    
                    # Force refresh to show the new messages
                    st.rerun()
                else:
                    st.warning("OpenAI API is not configured. Cannot generate response.")
                    st.session_state['doc_chat_history'].append({"role": "assistant", "content": "I'm sorry, I can't process your request because the OpenAI API is not configured. Please set up your API key to enable AI features."})
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state['doc_chat_history'].append({"role": "assistant", "content": "I'm sorry, I encountered an error processing your request. Please try again later."})
                st.rerun()

