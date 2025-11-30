import { Routes, Route } from "react-router-dom"
import Welcome from "./components/Welcome"
import AgentResult from "./components/AgentResult"
import Gallery from "./components/Gallery"
import Detail from "./components/Detail"
import Compare from "./components/Compare"

function App() {
  return (
    <Routes>
      {/* 첫 화면: 에이전트 입력 */}
      <Route path="/" element={<Welcome />} />

      {/* 에이전트 결과 */}
      <Route path="/agent" element={<AgentResult />} />

      {/* 기존 갤러리 / 상세 / 비교 기능 */}
      <Route path="/gallery" element={<Gallery />} />
      <Route path="/detail/:id" element={<Detail />} />
      <Route path="/compare" element={<Compare />} />
    </Routes>
  )
}

export default App
