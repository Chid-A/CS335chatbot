let currentLat = 48.8566;
let currentLon = 2.3522;

let selectedVoice = null;

// Load voices and pick a friendly female English voice
function loadVoices() {
  const voices = window.speechSynthesis.getVoices();
  console.log("Available voices:", voices.map(v => v.name));

  // Filter female English voices
  const femaleVoices = voices.filter(voice =>
    voice.lang.startsWith('en') &&
    (voice.name.toLowerCase().includes('female') ||
      voice.name.toLowerCase().includes('woman') ||
      voice.name.toLowerCase().includes('zira') ||      // Windows female voice
      voice.name.toLowerCase().includes('samantha') ||  // Mac female voice
      voice.name.toLowerCase().includes('google uk english female') || // Chrome female voice
      voice.name.toLowerCase().includes('google us english female'))
  );

  if (femaleVoices.length > 0) {
    selectedVoice = femaleVoices[0];
  } else {
    // fallback to any English voice
    selectedVoice = voices.find(voice => voice.lang.startsWith('en')) || voices[0];
  }
}

// Some browsers load voices asynchronously
window.speechSynthesis.onvoiceschanged = loadVoices;
setTimeout(loadVoices, 500);

function speakText(text) {
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }
  const utterance = new SpeechSynthesisUtterance(text);

  if (selectedVoice) {
    utterance.voice = selectedVoice;
  }

  // Slightly higher pitch and normal rate for a friendly tone
  utterance.pitch = 1.1;
  utterance.rate = 1;

  window.speechSynthesis.speak(utterance);
}

async function getBotResponse(userMessage) {
  try {
    const response = await fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        latitude: currentLat,
        longitude: currentLon,
      }),
    });
    const data = await response.json();
    currentLat = data.latitude;
    currentLon = data.longitude;

    return data.reply;
  } catch (error) {
    console.error("Backend error:", error);
    return "⚠️ Could not reach the server.";
  }
}

document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const input = document.getElementById("user-input");
  const userMessage = input.value.trim();
  if (!userMessage) return;

  addMessageToChat(userMessage, "user-message");
  input.value = "";

  // Add typing indicator
  const typingDiv = document.createElement("div");
  typingDiv.className = "bot-message typing-indicator";
  typingDiv.textContent = "👩 Typing...";
  const chatBox = document.getElementById("chat-box");
  chatBox.appendChild(typingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  // Simulate delay (1.2 - 2 seconds)
  const delay = Math.random() * (2000 - 1200) + 1200;
  await new Promise((resolve) => setTimeout(resolve, delay));

  const botReply = await getBotResponse(userMessage);

  // Remove typing indicator
  chatBox.removeChild(typingDiv);

  // Add actual bot reply
  addMessageToChat("👩 " + botReply, "bot-message");

  // Speak response
  speakText(botReply);
});

function addMessageToChat(message, className) {
  const chatBox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");
  messageDiv.className = className;
  messageDiv.textContent = message;
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

