import axios from "axios";

export const stylizeImage = async (face, style) => {
  const formData = new FormData();
  formData.append("face", face);
  formData.append("style", style);

  try {
    const res = await axios.post("http://localhost:8000/stylize", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res.data.result;
  } catch (err) {
    console.error("Stylization failed:", err);
    return null;
  }
};