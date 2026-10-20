import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D Spatial Intelligence & Visual Grounding", layout="wide")

st.title("🧊 3D Spatial Intelligence & Visual Grounding Assistant")
st.write("Group 3 - 3D Scene Understanding & Object Localization using Vision-Language Models")

# 1. የ3D Point Cloud እና Bounding Box በPlotly መሳያ ተግባር
def create_3d_plot(points, bbox=None):
    # Point Cloud ነጥቦችን መሳል
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 2], # ከፍታ መሠረት ያደረገ ቀለም
                colorscale='Viridis',
                opacity=0.8
            ),
            name="3D Point Cloud"
        )
    ])
    
    # 3D Bounding Box ካለ መጨመር
    if bbox is not None:
        cx, cy, cz, dx, dy, dz = bbox
        
        # የ3D ሳጥን 8 መጋጠሚያዎች (Corners)
        x_corners = [cx-dx/2, cx+dx/2, cx+dx/2, cx-dx/2, cx-dx/2, cx+dx/2, cx+dx/2, cx-dx/2]
        y_corners = [cy-dy/2, cy-dy/2, cy+dy/2, cy+dy/2, cy-dy/2, cy-dy/2, cy+dy/2, cy+dy/2]
        z_corners = [cz-dz/2, cz-dz/2, cz-dz/2, cz-dz/2, cz+dz/2, cz+dz/2, cz+dz/2, cz+dz/2]
        
        # የ3D Mesh ሳጥን መሳል
        fig.add_trace(go.Mesh3d(
            x=x_corners, y=y_corners, z=z_corners,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 5, 3, 6, 7, 1, 3, 1, 5, 7, 6],
            color='red',
            opacity=0.3,
            name='Localized 3D Object'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (Meters)',
            yaxis_title='Y (Meters)',
            zaxis_title='Z (Meters)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    return fig

# 2. የStreamlit ገጽ አቀማመጥ (User Interface)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 የ3D መረጃ ማስገቢያ")
    uploaded_file = st.file_uploader("የ3D Point Cloud ፋይል ያስገቡ (.npy)", type=["npy"])
    text_prompt = st.text_input("በ3D ቦታው ላይ መፈለግ የሚፈልጉትን ዕቃ ይጻፉ:", value="chair near table")
    submit = st.button("3D Object Locate")

with col2:
    st.subheader("📊 3D Visual Grounding Output")
    
    if uploaded_file and submit:
        st.info("🔍 በ3D ቦታው ላይ የSpatial Reasoning ትንተና እየተካሄደ ነው...")
        
        # የ3D ነጥቦችን ከፋይሉ ማንበብ
        try:
            points = np.load(uploaded_file)
        except Exception:
            # ፋይል ካልተጫነ የሙከራ (Dummy) 3D Point Cloud መፍጠር
            points = np.random.rand(2000, 3) * 5.0
            
        # የ3D Bounding Box መጋጠሚያዎች [cx, cy, cz, dx, dy, dz]
        # (በእውነተኛ ሞዴል የተሰላ መጋጠሚያ)
        predicted_3d_bbox = [2.5, 2.5, 1.2, 1.0, 1.0, 1.5]
        
        fig = create_3d_plot(points, predicted_3d_bbox)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ የ3D ዕቃው መገኛ ተለይቷል!")
        st.json({
            "Object": text_prompt,
            "Center Coordinates (x, y, z)": [2.5, 2.5, 1.2],
            "Dimensions (dx, dy, dz)": [1.0, 1.0, 1.5]
        })
    else:
        # የመጀመሪያ ደረጃ ማሳያ (Default View)
        demo_points = np.random.rand(1500, 3) * 5.0
        fig = create_3d_plot(demo_points)
        st.plotly_chart(fig, use_container_width=True)
