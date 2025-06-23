import React, { useState } from "react";
import { stylizeImage } from "../api";

const UploadForm = ({ setResultImage }) => {
  const [face, setFace] = useState(null);
  const [style, setStyle] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!face || !style) return;
    setLoading(true);
    const result = await stylizeImage(face, style);
    setResultImage(result);
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>Face Image: </label>
        <input type="file" accept="image/*" onChange={(e) => setFace(e.target.files[0])} />
      </div>
      <div>
        <label>Style Image: </label>
        <input type="file" accept="image/*" onChange={(e) => setStyle(e.target.files[0])} />
      </div>
      <button type="submit">Stylize</button>
      {loading && <p>Generating stylized image...</p>}
    </form>
  );
};

export default UploadForm;