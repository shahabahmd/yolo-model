import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [output, setOutput] = useState(null);
  const [summary, setSummary] = useState(null);
  const [model, setModel] = useState("soil");
  const [loading, setLoading] = useState(false);
  const [insight, setInsight] = useState("");

  // === Predefined Insights (Only for Soil Types) ===
  const insights = {
    clay: " Detected Clay Soil — found in low-drainage areas. Excellent for rice cultivation but poor for aeration.",
    alluvial: " Detected Alluvial Soil — fertile and found in river basins. Great for wheat, sugarcane, and pulses.",
    black: " Detected Black Soil — rich in iron and moisture-retaining. Ideal for cotton and soybean cultivation.",
    red: " Detected Red Soil — low in nitrogen but good drainage. Suitable for crops like millet, potato, and pulses.",
  };

  // === Drag & Drop Handler ===
  const onDrop = useCallback((acceptedFiles) => {
    const selected = acceptedFiles[0];
    setFile(selected);
    setOutput(null);
    setSummary(null);
    setInsight("");
    if (selected) setPreview(URL.createObjectURL(selected));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
  });

  // === File Input Handler ===
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setOutput(null);
    setSummary(null);
    setInsight("");
    if (selected) setPreview(URL.createObjectURL(selected));
  };

  // === Upload & Detect ===
  const handleUpload = async () => {
    if (!file) {
      alert("Please upload or drag an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/predict/${model}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.result_image) {
        setOutput(`http://127.0.0.1:8000${data.result_image}`);
        setSummary(data.summary || null);

        // 🧠 Show insights only for SOIL detection
        if (model === "soil" && data.summary?.detected_classes?.length > 0) {
          const topClassRaw = data.summary.detected_classes[0]
            .toLowerCase()
            .trim();

          let topClass = "";
          if (topClassRaw.includes("clay")) topClass = "clay";
          else if (topClassRaw.includes("alluvial")) topClass = "alluvial";
          else if (topClassRaw.includes("black")) topClass = "black";
          else if (topClassRaw.includes("red")) topClass = "red";

          if (insights[topClass]) {
            setInsight(insights[topClass]);
          } else {
            setInsight(""); // hide if soil type not found
          }
        } else {
          setInsight(""); // hide for vegetation detection
        }
      } else {
        alert("No detection result returned!");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong while detecting!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Soil & Vegetation Detection</h1>
      <p className="subtitle">
        Upload or drag an image to detect soil or vegetation types
      </p>

      <div className="card">
        {/* === Model Dropdown === */}
        <select
          className="dropdown"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="soil">Soil Detection (YOLOv11)</option>
          <option value="vegetation">Vegetation Detection (YOLOv8)</option>
        </select>

        {/* === Dropzone === */}
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? "active" : ""}`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the image here ...</p>
          ) : (
            <p>Drag & Drop your image here, or click to browse</p>
          )}
        </div>

        <p className="or">OR</p>

        {/* === File Upload === */}
        <input type="file" accept="image/*" onChange={handleFileChange} />

        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Detecting..." : "Upload & Detect"}
        </button>

        {loading && <div className="loader"></div>}

        {/* === Image Section === */}
        <div className="image-container">
          {preview && (
            <div className="image-box hover-dark">
              <h3>Uploaded Image</h3>
              <img src={preview} alt="Preview" className="image" />
            </div>
          )}

          {output && (
            <div className="image-box hover-dark">
              <h3>Detection Result</h3>
              <img src={output} alt="Result" className="image" />
            </div>
          )}
        </div>

        {/* === Detection Summary === */}
        {summary && (
          <div className="summary-card">
            <h3>📊 Detection Summary</h3>
            {summary.detected_classes.length > 0 ? (
              <>
                <ul>
                  {Object.entries(summary.class_counts).map(([cls, count]) => (
                    <li key={cls}>
                      <strong>{cls}</strong> — {count} detected
                    </li>
                  ))}
                </ul>

                <div className="confidence-section">
                  {summary.detailed.map((item, index) => (
                    <div key={index} className="confidence-bar">
                      <span>{item.class}</span>
                      <div className="bar">
                        <div
                          className="fill"
                          style={{ width: `${item.confidence}%` }}
                        ></div>
                      </div>
                      <span className="conf">{item.confidence}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p>No objects detected.</p>
            )}
          </div>
        )}

        {/* === Insight Section (only for soil) === */}
        {insight && model === "soil" && (
          <div className="insight-box">
            <p>{insight}</p>
          </div>
        )}
      </div>

      <footer>
        © {new Date().getFullYear()} Soil & Vegetation Detection by Shahab
        Ahamed
      </footer>
    </div>
  );
}

export default App;
