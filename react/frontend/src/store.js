import { configureStore, createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from './api/axios'

// 예시: 사용자 불러오기
export const fetchUsers = createAsyncThunk('users/fetch', async () => {
    const res = await api.get('/users')
    return res.data
})

const userSlice = createSlice({
    name: 'users',
    initialState: { list: [], loading: false },
    reducers: {},
    extraReducers: (builder) => {
        builder
            .addCase(fetchUsers.pending, (state) => { state.loading = true })
            .addCase(fetchUsers.fulfilled, (state, action) => {
                state.loading = false
                state.list = action.payload
            })
            .addCase(fetchUsers.rejected, (state) => { state.loading = false })
    }
})

export const store = configureStore({
    reducer: { users: userSlice.reducer }
})
