document.addEventListener("DOMContentLoaded", () => {
  const chatLog = document.getElementById("chat-log");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const chatInputArea = document.querySelector(".chat-input-area");

  // O histórico do chat será mantido aqui no frontend
  let chatHistory = [];

  // Função para adicionar uma mensagem à interface do chat
  const addMessage = (sender, message) => {
    // Remove o indicador de "digitando..." se ele existir
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
      typingIndicator.parentElement.remove();
    }

    const messageElement = document.createElement("div");
    messageElement.classList.add("message", sender); // sender é 'user' ou 'ai'

    const contentElement = document.createElement("div");
    contentElement.classList.add("message-content");
    
    const pElement = document.createElement("p");
    pElement.textContent = message;
    contentElement.appendChild(pElement);
    
    messageElement.appendChild(contentElement);
    chatLog.appendChild(messageElement);

    // Rola para a mensagem mais recente
    chatLog.scrollTop = chatLog.scrollHeight;
  };

  // Função para mostrar o indicador "digitando..."
  const showTypingIndicator = () => {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", "ai");
    messageElement.innerHTML = `
      <div class="typing-indicator" id="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    `;
    chatLog.appendChild(messageElement);
    chatLog.scrollTop = chatLog.scrollHeight;
  };

  // Função principal para enviar a mensagem para o backend
  const sendMessage = async () => {
    const messageText = userInput.value.trim();
    if (messageText === "") return;

    // Adiciona a mensagem do usuário à interface e ao histórico
    addMessage("user", messageText);
    chatHistory.push({ role: "user", content: messageText });

    // Limpa o input e desabilita enquanto espera a resposta
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;

    // Mostra que o bot está "digitando"
    showTypingIndicator();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: messageText,
          history: chatHistory, // Envia o histórico atual
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const aiResponse = data.response;

      // Adiciona a resposta da IA à interface e ao histórico
      addMessage("ai", aiResponse);
      chatHistory.push({ role: "ai", content: aiResponse });

    } catch (error) {
      console.error("Erro ao comunicar com o servidor:", error);
      addMessage("ai", "Desculpe, não consigo me conectar ao meu cérebro agora. Tente novamente mais tarde.");
    } finally {
      // Reabilita a área de input
      userInput.disabled = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  };

  // Event Listeners
  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

  // Mensagem inicial do bot
  const addInitialMessage = () => {
      const welcomeMessage = "Olá! Eu sou o Tomás, seu assistente virtual. Como posso ajudar hoje?";
      addMessage("ai", welcomeMessage);
      chatHistory.push({ role: "ai", content: welcomeMessage });
      userInput.disabled = false;
      userInput.placeholder = "Escreva a sua mensagem...";
  };

  addInitialMessage();
});