import React, { useState } from "react";
import axios from "axios";

const styles = ["anime", "cartoon", "sketch"];

export default function App() {
  const [faceFile, setFaceFile] = useState(null);
  const [style, setStyle] = useState("anime");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!faceFile) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("face", faceFile);
    formData.append("style_name", style);

    try {
      const res = await axios.post("http://localhost:8000/stylize", formData);
      setResult("data:image/jpeg;base64," + res.data.result);
    } catch (err) {
      alert("Stylization failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <h1 className="text-3xl font-bold mb-6">One-Shot Face Stylizer</h1>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFaceFile(e.target.files[0])}
        className="mb-4"
      />

      <select
        value={style}
        onChange={(e) => setStyle(e.target.value)}
        className="mb-4 border p-2 rounded"
      >
        {styles.map((s) => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>

      <button
        onClick={handleUpload}
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        disabled={loading}
      >
        {loading ? "Stylizing..." : "Stylize"}
      </button>

      {result && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-2">Stylized Output:</h2>
          <img src={result} alt="Stylized output" className="rounded shadow-md max-w-sm" />
        </div>
      )}
    </div>
  );
}