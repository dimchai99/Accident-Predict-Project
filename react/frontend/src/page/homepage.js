import React, { useEffect, useState } from "react";
import { AppBar, Toolbar, Typography, Button, Menu, MenuItem, ListItemIcon, ListItemText, Fade, Divider, IconButton, Drawer, List, ListItem, ListItemButton } from "@mui/material";
import { Link, useNavigate, useLocation } from "react-router-dom";
import MenuIcon from '@mui/icons-material/Menu';
import factoryimage from "../image/factory.png";
import {Box} from "@mui/material";

export default function Homepage(){
    const [openDrawer, setOpenDrawer] = useState(false); // 드로어 상태 관리

    const toggleDrawer = (open) => (event) => {
        if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
            return;
        }
        setOpenDrawer(open);
    };

    const list = () => (
        <Box
            sx={{ width: 250 }}
            role="presentation"
            onClick={toggleDrawer(false)} // 메뉴 항목 클릭 시 드로어 닫기
            onKeyDown={toggleDrawer(false)}
        >
            <List>
                <ListItem disablePadding>
                    <ListItemButton>
                        <ListItemText primary="메뉴 1" />
                    </ListItemButton>
                </ListItem>
                <ListItem disablePadding>
                    <ListItemButton>
                        <ListItemText primary="메뉴 2" />
                    </ListItemButton>
                </ListItem>
                {/* 필요한 만큼 메뉴 항목 추가 */}
            </List>
        </Box>
    );
    return(
        <>
            <AppBar position="static" sx={{ backgroundColor: '#151651' }} >
                <Toolbar sx={{ display: "flex", justifyContent: "space-between", fontWeight:"bold", fontSize:"20px"}}>
                    <IconButton
                        edge="start"
                        color="inherit"
                        aria-label="menu"
                        onClick={toggleDrawer(true)}
                    >
                        <MenuIcon />
                    </IconButton>
                </Toolbar>
            </AppBar>
          <Box
              sx={{
                  backgroundImage: `url(${factoryimage})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  height: '500px'
              }}
          >

          </Box>
        </>
    )
}