import sys
sys.modules["torch.classes"] = None

# app.py
import streamlit as st
import os
import importlib
import Full_Prompt_new  # Ensure this file is in your project folder

# Custom CSS to style the page
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
        padding: 2rem;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
    }
    .subheader {
        font-size: 1.5rem;
        color: #333333;
    }
    .icon {
        font-size: 2rem;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        color: #777777;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown("<div class='header'>✨ RAGcipe Culinary Assistant ✨</div>", unsafe_allow_html=True)
    
    # User enters the query
    query = st.text_input("🍽️ Enter your culinary query:", "high protein beef dish")
    # 1) Button to retrieve recipe choices
    if st.button("🚀 Get Recipe Choices"):
        with st.spinner("⏳ Querying recipes... Please wait."):
            recipe_choices = Full_Prompt_new.get_recipe_choices(query)
            # Store in session state so we can display them below
            st.session_state.recipe_choices = recipe_choices
            st.session_state.user_query = query

    # 2) If we have recipes in session state, display them
    if "recipe_choices" in st.session_state and st.session_state.recipe_choices:
        recipe_choices = st.session_state.recipe_choices
        st.write("## Available Recipe Choices")

        # Build an HTML table with bigger columns & better styling
        table_html = """
        <style>
        table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 1em;
          table-layout: fixed;
        }
        th, td {
          padding: 12px;
          border: 1px solid #ddd;
          vertical-align: top;
          word-wrap: break-word;
        }
        th {
          font-weight: bold;
        }
        .col-recipe {
          width: 30%;
        }
        .col-url {
          width: 70%;
        }
        </style>
        <table>
          <colgroup>
            <col class="col-recipe" />
            <col class="col-url" />
          </colgroup>
          <thead>
            <tr>
              <th>Recipe Name</th>
              <th>Recipe URL</th>
            </tr>
          </thead>
          <tbody>
        """

        # Fill in the table rows
        for idx, recipe in enumerate(recipe_choices):
            name = recipe["name"]
            url = recipe["url"]
            table_html += f"""<tr>
              <td>{name}</td>
              <td><a href="{url}" target="_blank">{url}</a></td>
            </tr>
            """

        table_html += """</tbody>
        </table>
        """
        
        # Render the HTML table
        st.markdown(table_html, unsafe_allow_html=True)

        # 3) Form to let user pick a recipe after seeing the table
        with st.form("select_recipe_form"):
            st.write("### Select a Recipe to Generate a Detailed Response")
            
            # Create radio buttons labeled with "index: name"
            option_labels = [
                f"{idx+1}: {recipe['name']}" 
                for idx, recipe in enumerate(recipe_choices)
            ]
            selected_recipe_idx = st.radio(
                "Choose one recipe:", 
                options=list(range(len(recipe_choices))),
                format_func=lambda x: option_labels[x]
            )
            
            # Submit button to confirm selection
            submitted = st.form_submit_button("Generate Recipe Response")

        if submitted:
            selected_recipe = recipe_choices[selected_recipe_idx]

            with st.spinner("🥘 Mixing ingredients and machine learning..."):
                # 4) Process the chosen recipe to get the LLM response
                response = Full_Prompt_new.process_selected_recipe(
                    st.session_state.user_query,
                    selected_recipe
                )
                
                if isinstance(response, dict) and "answer" in response:
                    formatted_response = response["answer"]
                else:
                    formatted_response = str(response)
                
                st.subheader("🧂 Seasoned with AI, Served with Love")
                st.markdown(formatted_response, unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2025 RAGcipe Team - Powered by OpenAI & FairPrice Data</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
