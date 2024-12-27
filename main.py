# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import io

app = FastAPI(title="Data Visualization Recommendation API")

# Supported visualization libraries
VisualizationLibrary = Literal[
    "chartjs", 
    "plotly", 
    "d3", 
    "echarts", 
    "highcharts"
]

class LibraryConfig(BaseModel):
    library: VisualizationLibrary
    version: Optional[str] = "latest"
    theme: Optional[str] = "default"

class VisualizationRecommendation(BaseModel):
    chart_type: str
    columns: List[str]
    score: float
    explanation: str
    supported_libraries: List[str]

class VisualizationResponse(BaseModel):
    config: Dict[str, Any]
    library_specific_config: Dict[str, Any]
    data: Dict[str, Any]

def generate_chartjs_config(chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Chart.js specific configuration"""
    return {
        "type": chart_type,
        "data": {
            "labels": data.get("labels", []),
            "datasets": [{
                "label": data.get("title", ""),
                "data": data.get("values", []),
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {"beginAtZero": True}
            }
        }
    }

def generate_plotly_config(chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Plotly specific configuration"""
    return {
        "data": [{
            "type": chart_type,
            "x": data.get("labels", []),
            "y": data.get("values", []),
            "name": data.get("title", "")
        }],
        "layout": {
            "title": data.get("title", ""),
            "xaxis": {"title": data.get("xlabel", "")},
            "yaxis": {"title": data.get("ylabel", "")}
        }
    }

def generate_echarts_config(chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ECharts specific configuration"""
    return {
        "title": {"text": data.get("title", "")},
        "xAxis": {
            "type": "category",
            "data": data.get("labels", [])
        },
        "yAxis": {"type": "value"},
        "series": [{
            "type": chart_type,
            "data": data.get("values", [])
        }]
    }

def generate_highcharts_config(chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Highcharts specific configuration"""
    return {
        "chart": {"type": chart_type},
        "title": {"text": data.get("title", "")},
        "xAxis": {
            "categories": data.get("labels", [])
        },
        "series": [{
            "name": data.get("title", ""),
            "data": data.get("values", [])
        }]
    }

def generate_d3_config(chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate D3.js specific configuration"""
    return {
        "data": [
            {"label": label, "value": value}
            for label, value in zip(
                data.get("labels", []),
                data.get("values", [])
            )
        ],
        "margin": {"top": 20, "right": 20, "bottom": 30, "left": 40},
        "width": 600,
        "height": 400
    }

LIBRARY_GENERATORS = {
    "chartjs": generate_chartjs_config,
    "plotly": generate_plotly_config,
    "echarts": generate_echarts_config,
    "highcharts": generate_highcharts_config,
    "d3": generate_d3_config
}

@app.post("/visualize", response_model=VisualizationResponse)
async def generate_visualization(
    file: UploadFile = File(...),
    library_config: LibraryConfig = Form(...),
    chart_type: str = Form(...),
    columns: str = Form(...),
    aggregation: Optional[str] = Form("mean"),
    title: Optional[str] = Form(None),
    theme: Optional[str] = Form("default")
):
    """Generate visualization configuration for specified library"""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        selected_columns = json.loads(columns)

        # Prepare data based on chart type and aggregation
        if chart_type in ["bar", "line"]:
            if len(selected_columns) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Bar and line charts require exactly 2 columns"
                )

            x_col, y_col = selected_columns

            if aggregation == "mean":
                data = df.groupby(x_col)[y_col].mean()
            elif aggregation == "sum":
                data = df.groupby(x_col)[y_col].sum()
            elif aggregation == "count":
                data = df.groupby(x_col)[y_col].count()

            viz_data = {
                "labels": data.index.tolist(),
                "values": data.values.tolist(),
                "title": title or f"{aggregation.capitalize()} of {y_col} by {x_col}",
                "xlabel": x_col,
                "ylabel": y_col
            }

        elif chart_type == "scatter":
            if len(selected_columns) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Scatter plots require exactly 2 columns"
                )

            x_col, y_col = selected_columns
            viz_data = {
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "title": title or f"Scatter plot of {y_col} vs {x_col}",
                "xlabel": x_col,
                "ylabel": y_col
            }

        elif chart_type == "histogram":
            if len(selected_columns) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="Histogram requires exactly 1 column"
                )

            col = selected_columns[0]
            hist, bins = np.histogram(df[col], bins='auto')
            viz_data = {
                "labels": [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)],
                "values": hist.tolist(),
                "title": title or f"Distribution of {col}",
                "xlabel": col,
                "ylabel": "Frequency"
            }

        # Generate library-specific configuration
        generator = LIBRARY_GENERATORS.get(library_config.library)
        if not generator:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported visualization library: {library_config.library}"
            )

        library_config = generator(chart_type, viz_data)

        # Add theme configuration if specified
        if theme != "default":
            library_config["theme"] = theme

        return VisualizationResponse(
            config={
                "chart_type": chart_type,
                "columns": selected_columns,
                "aggregation": aggregation,
                "title": viz_data["title"]
            },
            library_specific_config=library_config,
            data=viz_data
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/supported-libraries")
async def get_supported_libraries():
    """Get information about supported visualization libraries"""
    return {
        "chartjs": {
            "versions": ["2.9.4", "3.x"],
            "themes": ["default", "dark"],
            "supported_charts": ["bar", "line", "scatter", "pie", "doughnut"]
        },
        "plotly": {
            "versions": ["2.x", "latest"],
            "themes": ["default", "dark", "light"],
            "supported_charts": ["bar", "line", "scatter", "histogram", "box"]
        },
        "echarts": {
            "versions": ["4.x", "5.x"],
            "themes": ["default", "dark", "light", "vintage"],
            "supported_charts": ["bar", "line", "scatter", "pie", "radar"]
        },
        "highcharts": {
            "versions": ["9.x", "latest"],
            "themes": ["default", "dark", "sand-signika"],
            "supported_charts": ["bar", "line", "scatter", "area", "column"]
        },
        "d3": {
            "versions": ["6.x", "7.x"],
            "themes": ["default"],
            "supported_charts": ["bar", "line", "scatter", "pie", "force"]
        }
    }

@app.get("/library-examples")
async def get_library_examples():
    """Get example configurations for each supported library"""
    return {
        "chartjs": {
            "basic_bar": generate_chartjs_config("bar", {
                "labels": ["A", "B", "C"],
                "values": [1, 2, 3],
                "title": "Example Bar Chart"
            })
        },
        "plotly": {
            "basic_bar": generate_plotly_config("bar", {
                "labels": ["A", "B", "C"],
                "values": [1, 2, 3],
                "title": "Example Bar Chart"
            })
        },
        # Add examples for other libraries...
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
