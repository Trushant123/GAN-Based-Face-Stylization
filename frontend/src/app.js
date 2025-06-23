import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import "./App.css";

function App() {
  const [resultImage, setResultImage] = useState(null);

  return (
    <div className="App">
      <h1>Face Stylization</h1>
      <UploadForm setResultImage={setResultImage} />
      {resultImage && (
        <div>
          <h2>Stylized Output</h2>
          <img src={`data:image/jpeg;base64,${resultImage}`} alt="stylized output" width="256" />
        </div>
      )}
    </div>
  );
}

export default App;
