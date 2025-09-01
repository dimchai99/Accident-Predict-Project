import axios from 'axios'

const api = axios.create({
    baseURL: '/api',   // 개발 중엔 proxy로, 배포에선 같은 서버에서 서빙됨
    headers: {
        'Content-Type': 'application/json'
    }
})

export default api
