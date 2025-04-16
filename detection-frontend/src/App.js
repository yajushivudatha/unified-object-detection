import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [category, setCategory] = useState("human");
  const [resultImage, setResultImage] = useState(null);

  const handleSubmit = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("category", category);

    try {
      const response = await axios.post("http://127.0.0.1:8000/detect/", formData, {
        responseType: "blob",
      });
      const imageUrl = URL.createObjectURL(response.data);
      setResultImage(imageUrl);
    } catch (error) {
      alert("Detection failed. Check backend console.");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Unified Object Detection</h2>
      <input type="file" onChange={(e) => setSelectedFile(e.target.files[0])} />
      <br /><br />
      <select onChange={(e) => setCategory(e.target.value)} value={category}>
        <option value="human">Human</option>
        <option value="vehicle">Vehicle</option>
        <option value="animal">Animal</option>
      </select>
      <br /><br />
      <button onClick={handleSubmit}>Detect</button>
      <br /><br />
      {resultImage && (
        <img src={resultImage} alt="Detected result" style={{ maxWidth: "100%" }} />
      )}
    </div>
  );
}

export default App;
